import os
import math
import torch
import datasets
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Sampler, Dataset
from transformers.trainer import Trainer, is_datasets_available
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import logging

from .modeling_utils import evaluate_generation, evaluate_perplexity

logger = logging.get_logger(__name__)


class ActivationBeaconTrainer(Trainer):
    def __init__(self, *args, model_args, file_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.file_logger = file_logger

    def compute_loss(self, model, inputs, return_outputs=False):
        if "retrieval_span" in inputs:
            self.model.memory._retrieval_span = inputs['retrieval_span'][0]
            inputs.pop("retrieval_span")

        outputs = super().compute_loss(model, inputs, return_outputs)

        if hasattr(self.model, "memory") and hasattr(self.model.memory, "_retrieval_span"):
            del self.model.memory._retrieval_span
            del self.model.memory._retrieval_condensing_ratios
        return outputs

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
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # Build the sampler.
        if self.args.group_by_stride is not None:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = self.train_dataset[self.args.length_column_name]
            else:
                lengths = None
            
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None

            return StrideGroupedSampler(
                # NOTE: multiply world size to get the total number of training instances across devices
                batch_size=self.args.train_batch_size * self.args.world_size,
                window=self.model.memory.beacon_window,
                stride=self.model.memory.beacon_stride,
                group=self.args.group_by_stride,
                sort=self.args.sort_by_stride,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return super()._get_train_sampler()
    
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



class StrideGroupedSampler(Sampler):
    """Group """

    def __init__(
        self,
        batch_size: int,
        window: int,
        stride: int,
        group: str,
        sort: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        
        if group is None:
            raise ValueError("Group cannot be None!")

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        indices = list(range(len(lengths)))

        # get number of strides for each data
        num_strides = []
        for length in lengths:
            num_stride = math.ceil((length - window) / stride) + 1
            num_strides.append(num_stride)

        indice_stride_pairs = list(zip(indices, num_strides))
        # NOTE: shuffle the indices in advance, otherwise the randomness may be lost when all num_strides are equal
        random.shuffle(indice_stride_pairs)

        # sort data according to the number of strides
        indice_stride_pairs = sorted(indice_stride_pairs, key=lambda x: x[1])

        # group data instances with the same number of strides into the same batch
        batches = []
        batch = []
        prev_num_stride = None
        for index, num_stride in indice_stride_pairs:
            if num_stride != prev_num_stride:
                # in strict mode, all instances in the batch are forced to have the same number of strides
                if group == "strict":
                    batch.clear()
                elif group == "relaxed":
                    pass
                else:
                    raise ValueError(f"Group method {group} must be in None, strict, relaxed!")

            batch.append(index)
            prev_num_stride = num_stride

            if len(batch) == batch_size:
                batches.append((batch.copy(), num_stride))
                batch.clear()

        if len(batch) and group == "relaxed":
            batches.append((batch.copy(), num_stride))

        if sort is None:
            random.shuffle(batches)
        elif sort == "ascend":
            batches = sorted(batches, key=lambda x: x[1])
        elif sort == "descend":
            batches = sorted(batches, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Sort method {sort} must be in None, ascend, descend!")

        batches = [x[0] for x in batches]
        self.indices = sum(batches, [])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
