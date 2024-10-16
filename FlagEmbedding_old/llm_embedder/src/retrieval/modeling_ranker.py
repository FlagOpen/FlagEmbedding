import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.get_logger(__name__)


class CrossEncoder(torch.nn.Module):
    def __init__(self, ranker, dtype:str="fp16", cache_dir=None, accelerator:Accelerator=None) -> None:
        super().__init__()
        logger.info(f"Loading tokenizer and model from {ranker}...")
        self.tokenizer = AutoTokenizer.from_pretrained(ranker, cache_dir=cache_dir)

        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        if accelerator is not None:
            device = accelerator.device
        else:
            device = torch.device("cpu")

        self.ranker = AutoModelForSequenceClassification.from_pretrained(ranker, num_labels=1, cache_dir=cache_dir, torch_dtype=dtype).to(device)

    def gradient_checkpointing_enable(self):
        self.ranker.gradient_checkpointing_enable()

    def forward(self, cross, batch_size, **kwds):
        output = self.ranker(**cross)
        scores = output.logits.view(batch_size, -1)
        loss = nn.functional.cross_entropy(scores, scores.new_zeros(scores.shape[0], dtype=torch.long))
        return {"loss": loss}

    @torch.no_grad()
    def rerank(self, cross, batch_size, key_mask=None, hits=None, **kwds):
        output = self.ranker(**cross)
        score = output.logits.view(batch_size, -1)
        # mask padded candidates
        if key_mask is not None:
            score = score.masked_fill(~key_mask.bool(), torch.finfo(score.dtype).min)

        score, indice = score.sort(dim=-1, descending=True)
        if hits is not None:
            score = score[:, :hits]
            indice = indice[:, :hits]

        # NOTE: set the indice to -1 so that this prediction is ignored when computing metrics
        indice[score == torch.finfo(score.dtype).min] = -1
        return score, indice

    def save_pretrained(self, output_dir: str, *args, **kwargs):
        self.tokenizer.save_pretrained(
            os.path.join(output_dir, "ranker"))
        self.ranker.save_pretrained(
            os.path.join(output_dir, "ranker"))
