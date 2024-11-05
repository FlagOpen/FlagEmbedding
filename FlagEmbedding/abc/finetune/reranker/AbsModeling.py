import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class AbsRerankerModel(ABC, nn.Module):
    """Abstract class of embedding model for training.

    Args:
        base_model: The base model to train on.
        tokenizer (AutoTokenizer, optional): The tokenizer to use. Defaults to ``None``.
        train_batch_size (int, optional): Batch size used for training. Defaults to ``4``.
    """
    def __init__(
        self,
        base_model: None,
        tokenizer: AutoTokenizer = None,
        train_batch_size: int = 4,
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Activates gradient checkpointing for the current model.
        """
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        """
        Enables the gradients for the input embeddings.
        """
        self.model.enable_input_require_grads(**kwargs)

    @abstractmethod
    def encode(self, features):
        """Abstract method of encode.

        Args:
            features (dict): Teatures to pass to the model.
        """
        pass

    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None, teacher_scores: Optional[Tensor] = None):
        """The computation performed at every call.

        Args:
            pair (Union[Dict[str, Tensor], List[Dict[str, Tensor]]], optional): The query-document pair. Defaults to ``None``.
            teacher_scores (Optional[Tensor], optional): Teacher scores of knowledge distillation. Defaults to None.

        Returns:
            RerankerOutput: Output of reranker model.
        """
        ranker_logits = self.encode(pair) # (batch_size * num, dim)
        if teacher_scores is not None:
            teacher_scores = torch.Tensor(teacher_scores)
            teacher_targets = teacher_scores.view(self.train_batch_size, -1)
            teacher_targets = torch.softmax(teacher_targets.detach(), dim=-1)

        if self.training:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
            loss = self.compute_loss(grouped_logits, target)
            if teacher_scores is not None:
                teacher_targets = teacher_targets.to(grouped_logits.device)
                # print(teacher_targets, torch.mean(torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1)))
                loss += - torch.mean(torch.sum(torch.log_softmax(grouped_logits, dim=-1) * teacher_targets, dim=-1))
        else:
            loss = None

        # print(loss)
        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_loss(self, scores, target):
        """Compute the loss.

        Args:
            scores (torch.Tensor): Computed scores.
            target (torch.Tensor): The target value.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        """Save the model.

        Args:
            output_dir (str): Directory for saving the model.
        """
        # self.model.save_pretrained(output_dir)
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, *args, **kwargs):
        """
        Save the tokenizer and model.
        """
        self.tokenizer.save_pretrained(*args, **kwargs)
        return self.model.save_pretrained(*args, **kwargs)
