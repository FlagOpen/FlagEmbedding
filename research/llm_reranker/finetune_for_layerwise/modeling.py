import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class BiEncoderModel(nn.Module):
    def __init__(self,
                 model: None,
                 tokenizer: AutoTokenizer = None,
                 train_batch_size: int = 4,
                 start_layer: int = 8
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size


        self.start_layer = start_layer


    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features):
        if features is None:
            return None
        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True,
                             cutoff_layers=list(range(self.start_layer, self.model.config.num_hidden_layers+1)))
        _, max_indices = torch.max(features['labels'], dim=1)
        predict_indices = max_indices - 1
        all_logits = outputs.logits
        all_scores = []
        for logits in all_logits:
            logits = [logits[i, predict_indices[i]] for i in range(logits.shape[0])]
            scores = torch.stack(logits, dim=0)
            all_scores.append(scores.contiguous())
        return all_scores

    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None):
        ranker_logits = self.encode(pair) # (batch_size * num, dim)

        if self.training:
            loss = 0
            for logits in ranker_logits:
                grouped_logits = logits.view(self.train_batch_size, -1)
                target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
                loss += self.compute_loss(grouped_logits, target)

            teacher_scores = ranker_logits[-1].view(
                self.train_batch_size,
                -1
            )
            teacher_targets = torch.softmax(teacher_scores.detach(), dim=-1)
            for logits in ranker_logits[:-1]:
                student_scores = logits.view(
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
        # self.model.save_pretrained(output_dir)
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, **kwargs):
        self.tokenizer.save_pretrained(**kwargs)
        self.model.config.save_pretrained(**kwargs)
        return self.model.save_pretrained(**kwargs)


