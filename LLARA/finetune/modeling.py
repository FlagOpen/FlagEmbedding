import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
import torch.distributed as dist
from torch import nn, Tensor
from tqdm import trange, tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model: AutoModel = None,
                 tokenizer: AutoTokenizer = None,
                 normlized: bool = False,
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 sub_batch_size: int = -1
                 ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.temperature = temperature
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.sub_batch_size = sub_batch_size

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features):
        # input('continue?')
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
                    psg_out = self.model(**sub_features, return_dict=True, output_hidden_states=True)
                    ### modify
                    p_reps = psg_out.hidden_states[-1][:, -8:, :]
                    p_reps = torch.mean(p_reps, dim=1)
                    all_p_reps.append(p_reps)
                all_p_reps = torch.cat(all_p_reps, 0).contiguous()
                if self.normlized:
                    all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
                return all_p_reps.contiguous()
            else:
                psg_out = self.model(**features, return_dict=True, output_hidden_states=True)
                p_reps = psg_out.hidden_states[-1][:, -8:, :]
                p_reps = torch.mean(p_reps, dim=1)
                if self.normlized:
                    p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
                return p_reps.contiguous()
        else:
            all_p_reps = []
            for sub_features in features:
                psg_out = self.model(**sub_features, return_dict=True, output_hidden_states=True)
                ### modify
                p_reps = psg_out.hidden_states[-1][:, -8:, :]
                p_reps = torch.mean(p_reps, dim=1)
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
            if self.normlized:
                all_p_reps = torch.nn.functional.normalize(all_p_reps, dim=-1)
            return all_p_reps.contiguous()


    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]= None, passage: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None):
        # torch.cuda.empty_cache()
        # if self.process_rank == 1:
        #     print(query)
        q_reps = self.encode(query) # (batch_size, dim)
        p_reps = self.encode(passage) # (batch_size * num, dim)

        if self.training:
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.compute_loss(scores, target) # 同批内除了正样本以外的均为负样本

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        # print(loss)
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t # 给当前进程的q和doc加上梯度,当前的q对其他的d，更新；当前的d对其他的q，更新
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
