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
        sub_batch_size: int = -1,
        kd_loss_type: str = 'kl_div',
        kd_loss_plus_normal_loss: bool = True,
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
        self.kd_loss_plus_normal_loss = kd_loss_plus_normal_loss
        self.kd_loss_type = kd_loss_type
    
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
    
    def get_local_score(self, q_reps, p_reps, all_scores):
        group_size = p_reps.size(0) // q_reps.size(0)
        indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size
        specific_scores = []
        for i in range(group_size):
            specific_scores.append(
                all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i]
            )
        return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)

    def compute_local_score(self, q_reps, p_reps):
        all_scores = self.compute_score(q_reps, p_reps)
        loacl_scores = self.get_local_score(q_reps, p_reps, all_scores)
        return loacl_scores
    
    def forward(
        self, 
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, 
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        q_reps = self.encode(queries) # (batch_size, dim)
        p_reps = self.encode(passages) # (batch_size * group_size, dim)
        group_size = p_reps.size(0) // q_reps.size(0)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()   # (batch_size, group_size)

                teacher_targets = F.softmax(teacher_scores, dim=-1)  # (batch_size, group_size)
                
                if no_in_batch_neg_flag:
                    local_scores = self.compute_local_score(q_reps, p_reps) # (batch_size, group_size)
                    
                    # compute kd loss
                    loss = self.distill_loss(self.kd_loss_type, teacher_targets, local_scores, group_size=group_size)
                    
                    # add normal loss if needed
                    if self.kd_loss_plus_normal_loss:
                        local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long) # (batch_size)
                        loss += self.compute_loss(local_scores, local_targets)
                else:
                    if self.negatives_cross_device:
                        cross_q_reps = self._dist_gather_tensor(q_reps) # (world_size * batch_size, dim)
                        cross_p_reps = self._dist_gather_tensor(p_reps) # (world_size * batch_size * group_size, dim)
                        cross_scores = self.compute_score(cross_q_reps, cross_p_reps)   # (world_size * batch_size, world_size * batch_size * group_size)
                        
                        # compute kd loss
                        if self.kd_loss_type == "kl_div":
                            student_scores = self.get_local_score(cross_q_reps, cross_p_reps, cross_scores) # (world_size * batch_size, group_size)
                            student_scores = student_scores[
                                q_reps.size(0)*self.process_rank : q_reps.size(0)*(self.process_rank+1)
                            ]   # (batch_size, group_size)
                            
                            loss = self.distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)
                        elif self.kd_loss_type == "m3_kd_loss":
                            cross_teacher_targets = self._dist_gather_tensor(teacher_targets)   # (world_size * batch_size, group_size)
                            
                            loss = self.distill_loss(self.kd_loss_type, cross_teacher_targets, cross_scores, group_size)
                        else:
                            raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")

                        # add normal loss if needed
                        if self.kd_loss_plus_normal_loss:
                            cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
                            cross_targets = cross_idxs * group_size # (world_size * batch_size)
                            loss += self.compute_loss(cross_scores, cross_targets)
                    else:
                        scores = self.compute_score(q_reps, p_reps) # (batch_size, batch_size * group_size)
                        
                        # compute kd loss
                        if self.kd_loss_type == "kl_div":
                            student_scores = self.get_local_score(q_reps, p_reps, scores) # (batch_size, group_size)
                            
                            loss = self.distill_loss(self.kd_loss_type, teacher_targets, student_scores, group_size)
                        elif self.kd_loss_type == "m3_kd_loss":
                            loss = self.distill_loss(self.kd_loss_type, teacher_targets, scores, group_size)
                        else:
                            raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
                        
                        # add normal loss if needed
                        if self.kd_loss_plus_normal_loss:
                            idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                            targets = idxs * (p_reps.size(0) // q_reps.size(0)) # (batch_size)
                            loss += self.compute_loss(scores, targets)
            else:
                if no_in_batch_neg_flag:
                    local_scores = self.compute_local_score(q_reps, p_reps) # (batch_size, group_size)
                    local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long) # (batch_size)
                    loss = self.compute_loss(local_scores, local_targets)
                else:
                    if self.negatives_cross_device:
                        cross_q_reps = self._dist_gather_tensor(q_reps) # (world_size * batch_size, dim)
                        cross_p_reps = self._dist_gather_tensor(p_reps) # (world_size * batch_size * group_size, dim)
                        cross_scores = self.compute_score(cross_q_reps, cross_p_reps)   # (world_size * batch_size, world_size * batch_size * group_size)

                        cross_idxs = torch.arange(cross_q_reps.size(0), device=cross_q_reps.device, dtype=torch.long)
                        cross_targets = cross_idxs * group_size # (world_size * batch_size)
                        loss = self.compute_loss(cross_scores, cross_targets)
                    else:
                        scores = self.compute_score(q_reps, p_reps) # (batch_size, batch_size * group_size)
                        
                        idxs = torch.arange(q_reps.size(0), device=q_reps.device, dtype=torch.long)
                        targets = idxs * group_size # (batch_size)
                        loss = self.compute_loss(scores, targets)
        else:
            loss = None

        return EmbedderOutput(
            loss=loss,
        )

    @staticmethod
    def distill_loss(kd_loss_type, teacher_targets, student_scores, group_size):
        if kd_loss_type == 'kl_div':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, group_size) / (world_size * batch_size, group_size)
            return -torch.mean(torch.sum(teacher_targets * torch.log_softmax(student_scores, dim=-1), dim=-1))
        elif kd_loss_type == 'm3_kd_loss':
            # teacher_targets: (batch_size, group_size) / (world_size * batch_size, group_size)
            # student_scores: (batch_size, batch_size * group_size) / (world_size * batch_size, world_size * batch_size * group_size)
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
        else:
            raise ValueError(f"Invalid kd_loss_type: {kd_loss_type}")

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
