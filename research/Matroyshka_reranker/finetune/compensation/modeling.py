import copy
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)


def set_nested_attr(obj, attr, value):
    attributes = attr.split('.')
    for attribute in attributes[:-1]:
        obj = getattr(obj, attribute)
    setattr(obj, attributes[-1], value)


def get_nested_attr(obj, attr):
    attributes = attr.split('.')
    for attribute in attributes:
        obj = getattr(obj, attribute)
    return obj


class BiEncoderModel(nn.Module):
    def __init__(self,
                 model: None,
                 tmp_model: None,
                 tokenizer: AutoTokenizer = None,
                 compress_method: str = 'mean',
                 train_batch_size: int = 4,
                 cutoff_layers: List[int] = [2, 4],
                 compress_layers: List[int] = [6],
                 compress_ratios: List[int] = [2],
                 train_method: str = 'distill'
                 ):
        super().__init__()
        self.model = model
        self.tmp_model = tmp_model
        if self.tmp_model is not None:
            self.tmp_model_attrs = [i.replace('.weight', '') for i, _ in self.tmp_model.named_parameters()]

        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.pointCE = nn.BCEWithLogitsLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size
        self.compress_method = compress_method

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

        self.cutoff_layers = cutoff_layers
        self.compress_layers = compress_layers
        self.compress_ratios = compress_ratios
        self.train_method = train_method

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features, query_lengths, prompt_lengths):
        # input('continue?')
        if features is None:
            return None

        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True,
                             # compress_layer=random.choice(self.compress_layers),
                             # compress_layer=[random.choice([0, 1]) * i * 4 for i in range(7)],
                             compress_layer=[random.choice([0, 1]) * i for i in self.compress_layers],
                             compress_ratio=random.choice(self.compress_ratios),
                             cutoff_layers=self.cutoff_layers,
                             # cutoff_layers=random.choice([9, 12, 15, 18]),
                             query_lengths=query_lengths,
                             prompt_lengths=prompt_lengths)
        if self.config.layer_wise:
            scores = []
            for i in range(len(outputs.logits)):
                logits = last_logit_pool(outputs.logits[i], outputs.attention_masks[i])
                scores.append(logits)
        else:
            logits = last_logit_pool(outputs.logits, outputs.attention_masks)
            scores = logits[:, self.yes_loc]
        return scores

    def forward(self,
                pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
                query_lengths: List[int] = None,
                prompt_lengths: List[int] = None,
                teacher_scores: List[int] = None):
        # if dist.get_rank() == 0:
        #     print(self.tokenizer.decode(pair['input_ids'][0]))
        ranker_logits = self.encode(pair, query_lengths, prompt_lengths)  # (batch_size * num, dim)

        if self.training:
            if isinstance(ranker_logits, List):
                loss = 0

                for idx, logits in enumerate(ranker_logits[::-1]):
                    grouped_logits = logits.view(self.train_batch_size, -1)
                    target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
                    loss += self.compute_loss(grouped_logits, target)

            else:
                grouped_logits = ranker_logits.view(self.train_batch_size, -1)
                target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
                loss = self.compute_loss(grouped_logits, target)
                if self.train_method == 'distill':
                    teacher_scores = torch.tensor(teacher_scores, device=ranker_logits.device)
                    teacher_scores = teacher_scores.view(self.train_batch_size, -1)
                    teacher_targets = torch.softmax(teacher_scores.detach(), dim=-1)
                    student_scores = ranker_logits.view(
                        self.train_batch_size,
                        -1
                    )
                    loss += - torch.mean(torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1))

        else:
            loss = None

        # print(loss)
        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        if self.tmp_model is None:
            state_dict = self.model.state_dict()
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.tmp_model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, 'tmp_model.pth'))
            # torch.save(self.tmp_model, os.path.join(output_dir, 'tmp_model.pth'))

    def save_pretrained(self, **kwargs):
        if self.tmp_model is None:
            return self.model.save_pretrained(**kwargs)
        else:
            os.makedirs(kwargs['output_dir'], exist_ok=True)
            state_dict = self.tmp_model.state_dict()
            torch.save(state_dict, os.path.join(kwargs['output_dir'], 'tmp_model.pth'))
            return True
