import logging

import torch
from transformers import AutoModel, AutoTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel

logger = logging.getLogger(__name__)


class BiDecoderOnlyEmbedderICLModel(AbsEmbedderModel):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'last_token',
        normalize_embeddings: bool = False,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')

    def encode(self, features):
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_p_reps = []
                for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                    end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                    sub_features = {}
                    for k, v in features.items():
                        sub_features[k] = v[i:end_inx]
                    last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                    p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                    all_p_reps.append(p_reps)
                all_p_reps = torch.cat(all_p_reps, 0).contiguous()
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
            else:
                last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
                all_p_reps = self._sentence_embedding(last_hidden_state, features['attention_mask'])
                if self.normalize_embeddings:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
        else:
            all_p_reps = []
            for sub_features in features:
                last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                p_reps = self._sentence_embedding(last_hidden_state, sub_features['attention_mask'])
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
            if self.normalize_embeddings:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()

    def _sentence_embedding(self, last_hidden_state, attention_mask):
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "last_token":
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

    def compute_score(self, q_reps, p_reps):
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def _compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
