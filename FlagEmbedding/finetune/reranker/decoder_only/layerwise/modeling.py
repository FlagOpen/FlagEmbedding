import torch
from transformers import PreTrainedModel, AutoTokenizer
import logging
from typing import List, Union, Dict, Optional
from torch import Tensor

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel, RerankerOutput

logger = logging.getLogger(__name__)


class CrossDecoderModel(AbsRerankerModel):
    """
    Model class for decoder only reranker.

    Args:
        base_model (PreTrainedModel): The underlying pre-trained model used for encoding and scoring input pairs.
        tokenizer (AutoTokenizer, optional): The tokenizer for encoding input text. Defaults to ``None``.
        train_batch_size (int, optional): The batch size to use. Defaults to ``4``.
        start_layer (int, optional): Starting layer for layerwise. Defaults to ``8``.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: AutoTokenizer = None,
        train_batch_size: int = 4,
        start_layer: int = 8
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=train_batch_size,
        )

        self.start_layer = start_layer

    def encode(self, features):
        if features is None:
            return None
        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True)
        all_logits = outputs.logits
        all_scores = []
        for logits in all_logits:
            all_scores.append(logits[:, -1].contiguous())
        return all_scores

    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, teacher_scores: Optional[Tensor] = None):
        ranker_logits = self.encode(pair) # (batch_size * num, dim)

        if self.training:
            loss = 0
            for logits in ranker_logits:
                grouped_logits = logits.view(self.train_batch_size, -1)
                target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
                loss += self.compute_loss(grouped_logits, target)

            if teacher_scores is None:
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
                teacher_scores = torch.Tensor(teacher_scores)
                teacher_scores = teacher_scores.view(self.train_batch_size, -1)
                teacher_targets = torch.softmax(teacher_scores.detach(), dim=-1).to(ranker_logits[-1].device)
                for logits in ranker_logits:
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
