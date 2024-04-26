import os
import json
import math
import torch
import shutil
import datasets
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from typing import Union, Optional, Iterator, Tuple, Dict, List, Callable, Mapping
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import (
    is_datasets_available, 
    set_seed, 
    seed_worker, 
    enable_full_determinism,
    has_length,
    LengthGroupedSampler,
    RandomSampler
)
from transformers.utils import logging

from accelerate import Accelerator

from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR, LinearWarmupLR
from colossalai.nn.optimizer import HybridAdam

from .metrics import Metric
from .trainer import StrideGroupedSampler
from .modeling_utils import evaluate_generation, evaluate_perplexity, move_to_device
from .utils import format_numel_str, save_json, load_json


logger = logging.get_logger(__name__)



@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    total_time: float = 0.

    start_epoch: int = 0
    start_step: int = 0
    sample_start_index: int = 0

    log_history: List[Dict] = field(default_factory=lambda:[])

    def append_loss(self, loss: float):
        self.log_history.append({"global_step": self.global_step, "loss": loss})

    def append_metrics(self, metrics: Dict):
        if metrics is not None:
            self.log_history.append({"global_step": self.global_step, "metric": metrics})

    def get_metric(self):
        metric = None
        for record in reversed(self.log_history):
            if "global_step" not in record:
                metric = record
                break
        return metric

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))



class ColossalTrainer:
    def __init__(
        self, 
        model:Union[PreTrainedModel, nn.Module], 
        args:TrainingArguments, 
        data_collator:Optional[Callable]=None, 
        train_dataset:Optional[Union[Dataset, datasets.Dataset]]=None, 
        eval_dataset:Optional[Union[Dataset, datasets.Dataset]]=None, 
        tokenizer:Optional[PreTrainedTokenizer]=None, 
        compute_metrics:Optional[Callable]=None
    ) -> None:        
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics

        self.state = TrainerState()
        # create accelerator object
        self.accelerator = Accelerator(
            dispatch_batches=self.args.dispatch_batches,
            split_batches=self.args.split_batches,
        )
    
    @property
    def is_main_proc(self):
        return self.accelerator.process_index == 0

    def compute_loss(self, model, inputs):
        self.model.train()
        outputs = model(**inputs)
        if isinstance(outputs, Mapping):
            loss = outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        return loss

    def train(self):
        # ==============================
        # Post init for colossal
        # ==============================
        args = self.args

        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        
        args.num_train_epochs = int(args.num_train_epochs)
        if args.save_total_limit is not None and args.save_total_limit <= 2:
            args.save_total_limit = 2

        # ==============================
        # Initialize plugins
        # ==============================
        if args.colossal_plugin == "gemini":
            plugin = GeminiPlugin(precision=args.colossal_mp, initial_scale=2**16, max_norm=args.max_grad_norm)
        elif args.colossal_plugin == "gemini_auto":
            plugin = GeminiPlugin(precision=args.colossal_mp, placement_policy="auto", initial_scale=2**16, max_norm=args.max_grad_norm)
        elif args.colossal_plugin == "zero1":
            plugin = LowLevelZeroPlugin(
                stage=1, precision=args.colossal_mp, initial_scale=2**16, max_norm=args.max_grad_norm
            )
        elif args.colossal_plugin == "zero2":
            plugin = LowLevelZeroPlugin(
                stage=2, precision=args.colossal_mp, initial_scale=2**16, max_norm=args.max_grad_norm
            )
        elif args.colossal_plugin == "zero2_cpu":
            plugin = LowLevelZeroPlugin(
                stage=2, precision=args.colossal_mp, initial_scale=2**16, cpu_offload=True, max_norm=args.max_grad_norm
            )
        elif args.colossal_plugin == "hybrid_parallel":
            # modify the param accordingly, default configuration is for llama2-7b
            plugin = HybridParallelPlugin(
                tp_size=4,
                pp_size=2,
                num_microbatches=None,
                microbatch_size=1,
                enable_jit_fused=False,
                zero_stage=0,
                precision="fp32",
                initial_scale=1,
            )
        else:
            raise ValueError(f"Unknown plugin {args.colossal_plugin}")

        self.booster = Booster(plugin=plugin)
        
        # ==============================
        # Initialize dataloader
        # ==============================
        dataloader = self.get_train_dataloader()

        # ==============================
        # Resume if specified
        # ==============================

        # TODO: gradient-accumulation
        num_steps_per_epoch = len(dataloader)
        if args.max_steps > 0:
            num_train_epochs = math.ceil(args.max_steps / num_steps_per_epoch)
            max_steps = args.max_steps
        else:
            num_train_epochs = args.num_train_epochs
            max_steps = num_train_epochs * num_steps_per_epoch

        if args.save_strategy == "steps" and args.save_steps < 1:
            save_steps = math.floor(max_steps * args.save_steps)
        else:
            save_steps = args.save_steps

        if args.resume_from_checkpoint is not None:
            # TODO: resume from checkpoint
            raise NotImplementedError

            self.accelerator.print(f"Resume training from {args.resume_from_checkpoint}...")
            self.load_for_resume(model, optimizer, lr_scheduler, args.resume_from_checkpoint)
            self.accelerator.print(f"Loaded checkpoint {args.resume_from_checkpoint} at epoch {self.state.start_epoch} step {self.state.start_step}")

            max_steps = max_steps - self.state.start_step
            start_steps = num_train_epochs * self.state.start_epoch + self.state.start_step

            # if resume training, set the sampler start index to the correct value
            dataloader.sampler.set_start_index(self.state.sample_start_index)

        else:
            start_steps = 0

        # ==============================
        # Initialize optimizer and scheduler
        # ==============================
        model = self.model
        optimizer = HybridAdam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)

        warmup_steps = 0
        if args.warmup_ratio > 0:
            warmup_steps = math.ceil(max_steps * args.warmup_ratio)
        if args.warmup_steps > 0:
            warmup_steps = args.warmup_steps
        if args.lr_scheduler_type == "linear":
            lr_scheduler = LinearWarmupLR(
                optimizer, total_steps=max_steps, warmup_steps=warmup_steps
            )
        elif args.lr_scheduler_type == "cosine":
            lr_scheduler = CosineAnnealingWarmupLR(
                optimizer, total_steps=max_steps, warmup_steps=warmup_steps, eta_min=0.1 * args.learning_rate
            )
        else:
            raise NotImplementedError(f"Scheduler type {args.lr_scheduler_type} not implemented!")

        # ==============================
        # Boost model
        # ==============================
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        if args.colossal_mp == "fp16":
            default_dtype = torch.float16
        elif args.colossal_mp == "bf16":
            default_dtype = torch.bfloat16
        else:
            default_dtype = torch.float32
        torch.set_default_dtype(default_dtype)
        model, optimizer, _, dataloader, lr_scheduler = self.booster.boost(
            model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
        )
        torch.set_default_dtype(torch.float)

        use_pipeline = isinstance(self.booster.plugin, HybridParallelPlugin) and self.booster.plugin.pp_size > 1

        # ==============================
        # Train
        # ==============================
        if self.is_main_proc:
            logger.info(f"Model params: {format_numel_str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))}")

        pbar = tqdm(total=max_steps, initial=start_steps, disable=self.accelerator.process_index != 0)
        for epoch in range(self.state.start_epoch, num_train_epochs):
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)

            dataloader_iter = iter(dataloader)

            for step in range(1, num_steps_per_epoch + 1, 1):
                self.state.global_step += 1

                if use_pipeline:
                    outputs = self.booster.execute_pipeline(
                        dataloader_iter, model, lambda outputs, inputs: outputs.loss, optimizer, return_loss=True, return_outputs=True
                    )
                    loss = outputs["loss"]
                else:
                    batch = next(dataloader_iter)
                    inputs = move_to_device(batch, self.accelerator.device)
                    loss = self.compute_loss(model, inputs)
                    self.booster.backward(loss, optimizer)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if not use_pipeline:
                    # reduce and mean
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss.data
                    loss.div_(self.args.world_size)
                
                loss = round(loss.tolist(), 4)
                pbar.set_description(f"Epoch: {epoch},  Steps: {step},  Loss: {loss}")
                pbar.update(1)

                if args.logging_strategy == "steps" and (self.state.global_step) % args.logging_steps == 0:
                    self.accelerator.print({'loss': loss})
                    self.state.append_loss(loss)

                if args.evaluation_strategy == "steps" and (self.state.global_step) % save_steps == 0:
                    metrics = self.evaluate()
                    self.state.append_metrics(metrics)
                else:
                    metrics = None

                if args.save_strategy == "steps" and (self.state.global_step) % save_steps == 0:
                    self.save(model, optimizer, lr_scheduler, metrics=metrics)
                                
                if self.state.global_step >= max_steps:
                    break

            if args.evaluation_strategy == "epoch" and step == num_steps_per_epoch:
                metrics = self.evaluate()
                self.state.append_metrics(metrics)
            else:
                metrics = None

            if args.save_strategy == "epoch" and step == num_steps_per_epoch:
                self.save(model, optimizer, lr_scheduler, metrics=metrics)

    def evaluate(self):
        raise NotImplementedError
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )
        return self.accelerator.prepare(dataloader)

    def get_model_for_evaluation(self):
        model = self.accelerator.prepare_model(self.model, evaluation_mode=True).eval()
        return model

    def save(
        self,
        model: nn.Module,
        optimizer,
        lr_scheduler,
        metrics: Dict=None
    ):
        """Save model checkpoint and remove worse checkpoints."""
        save_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")

        if self.is_main_proc:
            logger.info(f"Saving checkpoint at {save_dir}...")
    
        # remove worse checkpoints
        if metrics is not None and self.args.save_total_limit is not None:
            if not hasattr(self, "_best_checkpoints"):
                checkpoints = []
                checkpoint_metrics = []
                
                for directory in os.listdir(self.args.output_dir):
                    abs_directory = os.path.join(self.args.output_dir, directory)
                    if os.path.isdir(abs_directory) and directory.startswith("checkpoint-") and os.path.exists(os.path.join(abs_directory, "states.json")):
                        metric = TrainerState.load_from_json(os.path.join(abs_directory, "states.json")).get_metric()
                        checkpoints.append(abs_directory)
                        checkpoint_metrics.append(metric)
                self._best_checkpoints = [(c, m[self.args.metric_for_best_model]) for c, m in zip(checkpoints, checkpoint_metrics)]

            # acendingly sort checkpoints by their metrics
            best_checkpoints = sorted(self._best_checkpoints, key=lambda x: x[1])
            
            if self.args.greater_is_better:
                checkpoints_to_remove = best_checkpoints[:-(self.args.save_total_limit - 1)]
                self._best_checkpoints = best_checkpoints[-(self.args.save_total_limit - 1):]
            else:
                checkpoints_to_remove = best_checkpoints[self.args.save_total_limit - 1:]
                self._best_checkpoints = best_checkpoints[:self.args.save_total_limit - 1]
            
            if self.is_main_proc:
                for checkpoint_to_remove in checkpoints_to_remove:
                    shutil.rmtree(checkpoint_to_remove[0])
            
            self._best_checkpoints.append((save_dir, metrics[self.args.metric_for_best_model]))

        self.accelerator.wait_for_everyone()

        if self.is_main_proc:
            os.makedirs(os.path.join(save_dir), exist_ok=True)

        # 4GB for 1 shard
        self.booster.save_model(model, save_dir, shard=True, size_per_shard=4096)
        if not self.args.save_only_model:
            self.booster.colossal_save_optim(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
            self.booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
        if self.is_main_proc:
            # NOTE: also save tokenizer
            self.tokenizer.save_pretrained(save_dir)
            self.state.save_to_json(os.path.join(save_dir, "states.json"))
        self.accelerator.wait_for_everyone()

    def load_for_resume(
        self, model: nn.Module, optimizer, lr_scheduler, load_dir
    ) -> Tuple[int, int, int]:
        """Load model checkpoint and optimizer."""
        self.booster.load_model(model, os.path.join(load_dir, "model"))
        self.booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
        self.booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
        self.state = TrainerState.load_from_json(os.path.join(load_dir, "states.json"))



class ColossalActivationBeaconTrainer(ColossalTrainer):
    def __init__(self, *args, model_args, file_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.file_logger = file_logger

    def compute_loss(self, model, inputs):
        # NOTE: reset memory for each individual input
        if hasattr(self.model, "memory"):
            self.model.memory.reset()

        inputs.pop("length", None)
        inputs.pop("index", None)

        outputs = super().compute_loss(model, inputs)

        if hasattr(self.model, "memory") and hasattr(self.model.memory, "_retrieval_span"):
            del self.model.memory._retrieval_span
            del self.model.memory._retrieval_condensing_ratios

        return outputs

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

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return

        dataloader = self.get_eval_dataloader()

        self.model.memory.reset()
        train_beacon_ratio = self.model.memory.beacon_ratio
        self.model.memory.set(beacon_ratio=self.args.eval_beacon_ratio)

        model = self.get_model_for_evaluation()

        if self.args.eval_method == "perplexity":
            perplexity = evaluate_perplexity(model, dataloader, accelerator=self.accelerator)
            metrics = {"perplexity": perplexity}
        elif self.args.eval_method == "generation":
            indices, outputs = evaluate_generation(
                model,
                dataloader, 
                accelerator=self.accelerator, 
                tokenizer=self.tokenizer,
                max_new_tokens=self.model_args.max_new_tokens,
                do_sample=self.model_args.do_sample,
                temperature=self.model_args.temperature,
                top_k=self.model_args.top_k,
                top_p=self.model_args.top_p,
            )
            metrics = self.compute_metrics(indices, outputs)

            if self.args.process_index == 0:
                result_path = Metric._get_save_path(self.model_args.eval_data, self.args.output_dir)
                Metric._save_result(indices=indices, preds=outputs, path=result_path, data=self.model_args.eval_data)
        else:
            raise NotImplementedError(f"Eval method {self.args.eval_method} not implemented!")

        self.model.memory.reset()
        self.model.memory.set(beacon_ratio=train_beacon_ratio)

        # log to file
        if self.args.process_index == 0:
            self.file_logger.log(
                metrics=metrics,
                Model_Args=asdict(self.model_args),
                Training_Args=asdict(self.args),
                Global_Steps=self.state.global_step
            )

        return metrics
