import os
import torch
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
from transformers.trainer import Trainer
from transformers.utils import logging

from .modeling_utils import evaluate_generation, evaluate_perplexity

logger = logging.get_logger(__name__)


class LLMTrainer(Trainer):
    def __init__(self, *args, model_args, file_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.file_logger = file_logger

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs.pop("length", None)
        inputs.pop("index", None)
        # move to GPU
        inputs = self._prepare_input(inputs)
        # NOTE: reset memory for each individual input
        if hasattr(self.model, "memory"):
            self.model.memory.reset()
        return inputs
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        outputs = super()._save(output_dir, state_dict)
        # NOTE: also save model_args
        self.model_args.save(os.path.join(output_dir, "model_args.json"))
        return outputs

    @torch.no_grad()
    def evaluate(self, eval_dataset: Dataset | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return

        if self.args.eval_method == "generation":
            labels = self.eval_dataset["labels"]
            self.eval_dataset = self.eval_dataset.remove_columns(["labels"])

        dataloader = self.get_eval_dataloader()

        self.model.memory.reset()
        train_beacon_ratio = self.model.memory.beacon_ratio
        train_beacon_ratio_mix = self.model.memory.beacon_ratio_mix
        self.model.memory.set(
            beacon_ratio=self.args.eval_beacon_ratio,
            beacon_ratio_mix=self.args.eval_beacon_ratio_mix,
        )

        model = self.model.eval()

        if self.args.eval_method == "perplexity":
            perplexity = evaluate_perplexity(model, dataloader, accelerator=self.accelerator)
            metrics = {"perplexity": perplexity}
        elif self.args.eval_method == "generation":
            indices, outputs = evaluate_generation(
                model, 
                dataloader, 
                accelerator=self.accelerator, 
                tokenizer=self.tokenizer,
            )
            metrics = self.compute_metrics(outputs, labels, indices=indices)
        else:
            raise NotImplementedError(f"Eval method {self.args.eval_method} not implemented!")

        self.model.memory.reset()
        self.model.memory.set(
            beacon_ratio=train_beacon_ratio,
            beacon_ratio_mix=train_beacon_ratio_mix,
        )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_") and key != "epoch":
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        # log to file
        if self.args.process_index == 0:
            self.file_logger.log(
                metrics=metrics,
                Model_Args=asdict(self.model_args),
                Training_Args=asdict(self.args),
                Global_Steps=self.state.global_step
            )

        return metrics