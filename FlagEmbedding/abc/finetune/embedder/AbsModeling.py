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

logger = logging.getLogger(__name__)


@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class AbsEmbedderModel(ABC, nn.Module):
    def __init__(
        self,
        base_model,
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer

        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.sub_batch_size = sub_batch_size
    
    @abstractmethod
    def encode(self, features):
        pass
    
    @abstractmethod
    def compute_loss(self, scores, target):
        pass
    
    @abstractmethod
    def compute_score(self, q_reps, p_reps):
        pass
    
    @abstractmethod
    def save(self, output_dir: str):
        pass
    
    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, 
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * num, dim)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()

                teacher_targets = F.softmax(teacher_scores, dim=-1)  # B N
                group_size = p_reps.size(0) // q_reps.size(0)

                if self.negatives_cross_device and not no_in_batch_neg_flag:
                    cross_q_reps = self._dist_gather_tensor(q_reps)
                    cross_p_reps = self._dist_gather_tensor(p_reps)
                    cross_teacher_targets = self._dist_gather_tensor(teacher_targets)
                    cross_scores = self.compute_score(cross_q_reps, cross_p_reps)

                    loss = self.distill_loss(cross_teacher_targets, cross_scores, group_size=group_size)
                else:
                    scores = self.compute_score(q_reps, p_reps)  # B, B * N
                    loss = self.distill_loss(teacher_targets, scores, group_size=group_size)
            else:
                if self.negatives_cross_device and not no_in_batch_neg_flag:
                    cross_q_reps = self._dist_gather_tensor(q_reps)
                    cross_p_reps = self._dist_gather_tensor(p_reps)

                    cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)

                    cross_targets = cross_idxs * (cross_p_reps.size(0) // cross_q_reps.size(0))
                    cross_scores = self.compute_score(cross_q_reps, cross_p_reps)

                    loss = self.compute_loss(cross_scores, cross_targets)
                else:
                    idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                    targets = idxs * (p_reps.size(0) // q_reps.size(0))

                    scores = self.compute_score(q_reps, p_reps)  # B, B * N
                    loss = self.compute_loss(scores, targets)
        else:
            loss = None

        return EmbedderOutput(
            loss=loss,
        )

    def distill_loss(self, teacher_targets, student_scores, group_size):
        labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long)
        labels = labels * group_size

        loss = 0
        mask = torch.zeros_like(student_scores)
        for i in range(group_size):
            temp_target = labels + i
            temp_scores = student_scores + mask
            temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")  # B
            loss += torch.mean(teacher_targets[:, i] * temp_loss)
            mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                 value=torch.finfo(student_scores.dtype).min)
        return loss

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
