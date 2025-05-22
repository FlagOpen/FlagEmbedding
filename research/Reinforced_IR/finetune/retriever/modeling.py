import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

import torch
from transformers import AutoModel, AutoTokenizer

from FlagEmbedding.abc.finetune.embedder.AbsModeling import AbsEmbedderModel, EmbedderOutput
from FlagEmbedding.finetune.embedder.encoder_only.base.modeling import BiEncoderOnlyEmbedderModel

logger = logging.getLogger(__name__)


class BiIREmbedderModel(BiEncoderOnlyEmbedderModel):
    """Embedder class for encoder only model.

    Args:
        base_model (AutoModel): The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        negatives_cross_device (bool, optional): If True, will compute cross devices negative loss. Defaults to ``False``.
        temperature (float, optional): Temperature to control the scale of scores. Defaults to ``1.0``.
        sub_batch_size (int, optional): Sub-batch size during encoding. If negative, will not split to sub-batch.
            Defaults to ``-1``.
        kd_loss_type (str, optional): Type of knowledge distillation loss. Defaults to ``"kl_div"``.
        sentence_pooling_method (str, optional): Pooling method to get sentence embedding. Defaults to ``'cls'``.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to ``False``.
    """
    TRANSFORMER_CLS = AutoModel
    
    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        answer_temperature: float = None,
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        sentence_pooling_method: str = 'cls',
        normalize_embeddings: bool = False,
        normalize_answer: bool = True,
        training_type: str = 'retrieval_answer'
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
            sentence_pooling_method=sentence_pooling_method,
            normalize_embeddings=normalize_embeddings
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.training_type = training_type
        if answer_temperature is not None:
            self.answer_temperature = answer_temperature
        else:
            self.answer_temperature = 0.05
        self.normalize_answer = normalize_answer

    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        answers: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        teacher_scores_answers: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        """The computation performed at every call.

        Args:
            queries (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input queries. Defaults to ``None``.
            passages (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): Input passages. Defaults to ``None``.
            teacher_scores (Union[None, List[float]], optional): Teacher scores for distillation. Defaults to ``None``.
            no_in_batch_neg_flag (bool, optional): If True, use no in-batch negatives and no cross-device negatives. Defaults to ``False``.

        Returns:
            EmbedderOutput: Output of the forward call of model.
        """
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * group_size, dim)
        if 'answer' in self.training_type or 'passage' in self.training_type:
            a_reps = self.encode(answers)

        group_size = p_reps.size(0) // q_reps.size(0)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                # teacher_scores = (teacher_scores + 3) * 2
                # teacher_scores = - teacher_scores.reciprocal() / 0.2 # / self.temperature
                # print(teacher_scores)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()   # (batch_size, group_size)
                # print(teacher_scores)
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # (batch_size, group_size)
                # print(teacher_targets)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            loss = 0
            if self.normalize_answer:
                current_norm = torch.norm(a_reps, p=2, dim=1)
                mse_loss = F.mse_loss(current_norm, torch.full_like(current_norm, 1.0))
                loss += mse_loss
            if 'retrieval' in self.training_type:
                scores, q_loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
                loss += q_loss
            if 'answer' in self.training_type:
                tmp_temperature = self.temperature
                self.temperature = self.answer_temperature
                _, a_loss = compute_loss_func(q_reps, a_reps)
                
                self.temperature = tmp_temperature
                loss += a_loss
                # if self.process_rank == 0:
                #     print('The norm of queries:', torch.norm(q_reps, dim=1).tolist()[:10])
                #     print('The norm of answer:', torch.norm(a_reps, dim=1).tolist()[:10])
                #     print('query passage scores:', torch.matmul(q_reps, p_reps.t())[:10])
                #     print('answer passage scores:', torch.matmul(a_reps, p_reps.t())[:10])

            if 'passage' in self.training_type:
                _, p_loss = compute_loss_func(a_reps, p_reps)
                loss += 0.25 * p_loss
                # if self.process_rank == 0:
                #     print('The norm of queries:', torch.norm(q_reps, dim=1).tolist()[:10])
                #     print('The norm of answer:', torch.norm(a_reps, dim=1).tolist()[:10])
                #     print('query passage scores:', torch.matmul(q_reps, p_reps.t())[:10])
                #     print('answer passage scores:', torch.matmul(a_reps, p_reps.t())[:10])
        else:
            loss = None

        return EmbedderOutput(
            loss=loss,
        )

    @staticmethod
    def distill_loss(kd_loss_type, teacher_targets, student_scores, group_size=None):
        """Compute the distillation loss.

        Args:
            kd_loss_type (str): Type of knowledge distillation loss, supports "kl_div" and "m3_kd_loss".
            teacher_targets (torch.Tensor): Targets from the teacher model.
            student_scores (torch.Tensor): Score of student model.
            group_size (int, optional): Number of groups for . Defaults to ``None``.

        Raises:
            ValueError: Invalid kd_loss_type

        Returns:
            torch.Tensor: A scalar of computed distillation loss.
        """
        if kd_loss_type == 'kl_div':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, group_size) / (world_size * batch_size, group_size)
            return - torch.mean(
                torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1)
            )
        elif kd_loss_type == 'm3_kd_loss':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, batch_size * group_size) / (world_size * batch_size, world_size * batch_size * group_size)
            labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long)
            labels = labels * group_size

            loss = 0
            mask = torch.zeros_like(student_scores)
            for i in range(group_size):
            # for i in range(2):
                temp_target = labels + i
                temp_scores = student_scores + mask
                # temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")  # B
                # loss += torch.mean(teacher_targets[:, i] * temp_loss)
                # print(teacher_targets[:, i])
                temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="mean")
                loss += temp_loss
                # break
                mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                    value=torch.finfo(student_scores.dtype).min)
            return loss / group_size
        else:
            raise ValueError(f"Invalid kd_loss_type: {kd_loss_type}")

    def save(self, output_dir: str):
        """Save the model to the directory.

        Args:
            output_dir (str): Directory for saving the model.
        """
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)