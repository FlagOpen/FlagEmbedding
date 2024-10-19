import os
import logging
from typing import Dict, List, Union, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel, EmbedderOutput

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderM3Model(AbsEmbedderModel):
    def __init__(
        self,
        base_model: Dict[str, Any],
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1,
        sub_batch_size: int = -1,
        sentence_pooling_method: str = 'cls',
        normalize_embeddings: bool = False,
        unified_finetuning: bool = True,
        use_self_distill: bool = False,
        self_distill_start_step: int = -1
    ):
        super().__init__(
            base_model,
            tokenizer,
            negatives_cross_device,
            temperature,
            sub_batch_size
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.unified_finetuning = unified_finetuning
        if not self.unified_finetuning:
            self.model = base_model['model']
            self.colbert_linear = None
            self.sparse_linear = None
        else:
            self.model = base_model['model']
            self.colbert_linear = base_model['colbert_linear']
            self.sparse_linear = base_model['sparse_linear']
        
        self.use_self_distill = use_self_distill
        self.self_distill_start_step = self_distill_start_step
        self.step = 0
    
    def _dense_embedding(self, last_hidden_state, attention_mask):
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "last":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                return last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device),
                    sequence_lengths,
                ]
        else:
            raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")

    def _sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding: return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def _colbert_embedding(self, last_hidden_state, mask):
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def compute_score(
        self, q_reps, p_reps, q_mask: torch.Tensor,
        dense_weight: float = 1.0, sparse_weight: float = 0.3, colbert_weight: float = 1.0
    ):
        dense_score = self.dense_score(q_reps, p_reps)
        sparse_score = self.sparse_score(q_reps, p_reps)
        colbert_score = self.colbert_score(q_reps, p_reps, q_mask=q_mask)
        return dense_score * dense_weight + sparse_score * sparse_weight + colbert_score * colbert_weight

    def dense_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def sparse_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def colbert_score(self, q_reps, p_reps, q_mask: torch.Tensor):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        scores = scores / self.temperature
        return scores

    def _encode(self, features):
        dense_vecs, sparse_vecs, colbert_vecs = None, None, None
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
        dense_vecs = self._dense_embedding(last_hidden_state, features['attention_mask'])
        if self.unified_finetuning:
            sparse_vecs = self._sparse_embedding(last_hidden_state, features['input_ids'])
            colbert_vecs = self._colbert_embedding(last_hidden_state, features['attention_mask'])
        if self.normalize_embeddings:
            dense_vecs = F.normalize(dense_vecs, dim=-1)
            if self.unified_finetuning:
                colbert_vecs = F.normalize(colbert_vecs, dim=-1)
        return dense_vecs, sparse_vecs, colbert_vecs

    def encode(self, features):
        if features is None:
            return None
        
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size != -1:
                all_dense_vecs, all_sparse_vecs, all_colbert_vecs = [], [], []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]

                    dense_vecs, sparse_vecs, colbert_vecs = self._encode(sub_features)
                    all_dense_vecs.append(dense_vecs)
                    all_sparse_vecs.append(sparse_vecs)
                    all_colbert_vecs.append(colbert_vecs)

                dense_vecs = torch.cat(all_dense_vecs, 0)
                if self.unified_finetuning:
                    sparse_vecs = torch.cat(all_sparse_vecs, 0)
                    colbert_vecs = torch.cat(all_colbert_vecs, 0)
            else:
                dense_vecs, sparse_vecs, colbert_vecs = self._encode(features)
        else:
            all_dense_vecs, all_sparse_vecs, all_colbert_vecs = [], [], []
            for sub_features in features:
                dense_vecs, sparse_vecs, colbert_vecs = self._encode(sub_features)
                all_dense_vecs.append(dense_vecs)
                all_sparse_vecs.append(sparse_vecs)
                all_colbert_vecs.append(colbert_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            if self.unified_finetuning:
                sparse_vecs = torch.cat(all_sparse_vecs, 0)
                colbert_vecs = torch.cat(all_colbert_vecs, 0)

        if self.unified_finetuning:
            return dense_vecs.contiguous(), sparse_vecs.contiguous(), colbert_vecs.contiguous()
        else:
            return dense_vecs.contiguous(), None, None

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, 
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        q_dense_vecs, q_sparse_vecs, q_colbert_vecs = self.encode(queries)
        p_dense_vecs, p_sparse_vecs, p_colbert_vecs = self.encode(passages)

        if self.training:
            if teacher_scores is not None:
                # print("Use soft-label distillation...")
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # B N
                group_size = p_dense_vecs.size(0) // q_dense_vecs.size(0)

                # dense loss
                dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
                if self.negatives_cross_device and not no_in_batch_neg_flag:
                    cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
                    cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)
                    cross_teacher_targets = self._dist_gather_tensor(teacher_targets)
                    cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

                    loss = self.distill_loss(cross_teacher_targets, cross_dense_scores, group_size=group_size)
                else:
                    loss = self.distill_loss(teacher_targets, dense_scores, group_size=group_size)

                if self.unified_finetuning:
                    # sparse and colbert loss
                    sparse_scores = self.sparse_score(q_sparse_vecs, p_sparse_vecs)  # B, B * N
                    sparse_loss = self.distill_loss(teacher_targets, sparse_scores, group_size=group_size)
                    
                    # padding attention mask for colbert
                    if not isinstance(queries, list):
                        q_mask = queries['attention_mask']
                    else:
                        q_mask_list = [sub_features['attention_mask'] for sub_features in queries]
                        _length = max([mask.shape[1] for mask in q_mask_list])
                        q_mask = torch.cat([
                            F.pad(mask, _length - mask.shape[1], value=0)
                            for mask in q_mask_list
                        ], dim=0)
                    
                    colbert_scores = self.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                        q_mask=q_mask)  # B, B * N
                    colbert_loss = self.distill_loss(teacher_targets, colbert_scores, group_size=group_size)

                    ensemble_loss = self.distill_loss(teacher_targets,
                                                      dense_scores + 0.3 * sparse_scores + colbert_scores,
                                                      group_size=group_size)
                    loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4

            else:
                idxs = torch.arange(q_dense_vecs.size(0), device=q_dense_vecs.device, dtype=torch.long)
                targets = idxs * (p_dense_vecs.size(0) // q_dense_vecs.size(0))

                # dense loss
                dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
                if self.negatives_cross_device and not no_in_batch_neg_flag:
                    cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
                    cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)

                    cross_idxs = torch.arange(cross_q_dense_vecs.size(0), device=cross_q_dense_vecs.device, dtype=torch.long)

                    cross_targets = cross_idxs * (cross_p_dense_vecs.size(0) // cross_q_dense_vecs.size(0))
                    cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

                    loss = self.compute_loss(cross_dense_scores, cross_targets)
                else:
                    loss = self.compute_loss(dense_scores, targets)

                if self.unified_finetuning:
                    # sparse and colbert loss
                    sparse_scores = self.sparse_score(q_sparse_vecs, p_sparse_vecs)  # B, B * N
                    sparse_loss = self.compute_loss(sparse_scores, targets)
                    
                    # padding attention mask for colbert
                    if not isinstance(queries, list):
                        q_mask = queries['attention_mask']
                    else:
                        q_mask_list = [sub_features['attention_mask'] for sub_features in queries]
                        _length = max([mask.shape[1] for mask in q_mask_list])
                        q_mask = torch.cat([
                            F.pad(mask, _length - mask.shape[1], value=0)
                            for mask in q_mask_list
                        ], dim=0)

                    colbert_scores = self.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                        q_mask=q_mask)  # B, B * N
                    colbert_loss = self.compute_loss(colbert_scores, targets)

                    ensemble_loss = self.compute_loss(dense_scores + 0.3 * sparse_scores + colbert_scores, targets)
                    loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4

            if self.use_self_distill and self.step > self.self_distill_start_step and self.unified_finetuning:
                ensemble_scores = dense_scores + 0.3 * sparse_scores + colbert_scores
                teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)
                ensemble_distill_dense_loss = - torch.mean(
                    torch.sum(torch.log_softmax(dense_scores, dim=-1) * teacher_targets, dim=-1))
                ensemble_distill_sparse_loss = - torch.mean(
                    torch.sum(torch.log_softmax(sparse_scores, dim=-1) * teacher_targets, dim=-1))
                ensemble_distill_colbert_loss = - torch.mean(
                    torch.sum(torch.log_softmax(colbert_scores, dim=-1) * teacher_targets, dim=-1))
                loss += (ensemble_distill_dense_loss + 0.1 * ensemble_distill_sparse_loss + ensemble_distill_colbert_loss) / 3
                loss = loss / 2
            self.step += 1
        else:
            loss = None
        return EmbedderOutput(
            loss=loss,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)
    
    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict

        self.model.save_pretrained(output_dir, state_dict=_trans_state_dict(self.model.state_dict()))

        if self.unified_finetuning:
            torch.save(_trans_state_dict(self.colbert_linear.state_dict()),
                       os.path.join(output_dir, 'colbert_linear.pt'))
            torch.save(_trans_state_dict(self.sparse_linear.state_dict()),
                       os.path.join(output_dir, 'sparse_linear.pt'))


class EncoderOnlyEmbedderM3ModelForInference(EncoderOnlyEmbedderM3Model):
    def forward(self,
                text_input: Dict[str, Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_colbert_vecs: bool = False,
                return_sparse_embedding: bool = False):
        assert return_dense or return_sparse or return_colbert_vecs, 'Must choose one or more from `return_colbert_vecs`, `return_sparse`, `return_dense` to set `True`!'

        last_hidden_state = self.model(**text_input, return_dict=True).last_hidden_state

        output = {}
        if return_dense:
            dense_vecs = self._dense_embedding(last_hidden_state, text_input['attention_mask'])
            output['dense_vecs'] = dense_vecs
        if return_sparse:
            sparse_vecs = self._sparse_embedding(last_hidden_state, text_input['input_ids'],
                                                return_embedding=return_sparse_embedding)
            output['sparse_vecs'] = sparse_vecs
        if return_colbert_vecs:
            colbert_vecs = self._colbert_embedding(last_hidden_state, text_input['attention_mask'])
            output['colbert_vecs'] = colbert_vecs

        if self.normalize_embeddings:
            if 'dense_vecs' in output:
                output['dense_vecs'] = F.normalize(output['dense_vecs'], dim=-1)
            if 'colbert_vecs' in output:
                output['colbert_vecs'] = F.normalize(output['colbert_vecs'], dim=-1)

        return output
