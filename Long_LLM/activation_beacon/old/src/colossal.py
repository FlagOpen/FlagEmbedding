import os
import json
import math
import random
import torch
import shutil
import datasets
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass, asdict, field
from typing import Union, Optional, Iterator, Tuple, Dict, List, Callable
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR, LinearWarmupLR
from colossalai.nn.optimizer import HybridAdam



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

        self.coordinator = DistCoordinator()
        self.state = TrainerState()
        self.accelerator = Accelerator()
    
    def compute_loss(self, model, inputs):
        return model(**inputs)

    def train(self):
        # ==============================
        # Post init for colossal
        # ==============================
        self.args.num_train_epochs = int(self.args.num_train_epochs)
        if self.args.save_total_limit is not None and self.args.save_total_limit <= 2:
            self.args.save_total_limit = 2

        # ==============================
        # Initialize plugins
        # ==============================
        if self.args.colossal_plugin == "gemini":
            plugin = GeminiPlugin(precision=self.args.mixed_precision, initial_scale=2**16, max_norm=self.args.max_grad_norm)
        elif self.args.colossal_plugin == "gemini_auto":
            plugin = GeminiPlugin(
                precision=self.args.mixed_precision, placement_policy="auto", initial_scale=2**16, max_norm=self.args.max_grad_norm
            )
        elif self.args.colossal_plugin == "zero1":
            plugin = LowLevelZeroPlugin(
                stage=1, precision=self.args.mixed_precision, initial_scale=2**16, max_norm=self.args.max_grad_norm
            )
        elif self.args.colossal_plugin == "zero2":
            plugin = LowLevelZeroPlugin(
                stage=2, precision=self.args.mixed_precision, initial_scale=2**16, max_norm=self.args.max_grad_norm
            )
        elif self.args.colossal_plugin == "zero2_cpu":
            plugin = LowLevelZeroPlugin(
                stage=2, precision=self.args.mixed_precision, initial_scale=2**16, cpu_offload=True, max_norm=self.args.max_grad_norm
            )
        elif self.args.colossal_plugin == "hybrid_parallel":
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
            raise ValueError(f"Unknown plugin {self.args.colossal_plugin}")

        self.booster = Booster(plugin=plugin)
        
        # ==============================
        # Initialize dataloader
        # ==============================
        dataloader = self.get_train_dataloader()

        # ==============================
        # Initialize optimizer and scheduler
        # ==============================
        model = self.model
        optimizer = HybridAdam(model.parameters(), lr=self.args.learning_rate, betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.weight_decay)

        total_steps = self.args.num_train_epochs * len(dataloader)
        warmup_steps = 0
        if self.args.warmup_ratio > 0:
            warmup_steps = math.ceil(total_steps * 0.03)
        if self.args.warmup_steps > 0:
            warmup_steps = self.args.warmup_steps
        if self.args.lr_scheduler_type == "linear":
            lr_scheduler = LinearWarmupLR(
                optimizer, total_steps=total_steps, warmup_steps=warmup_steps
            )
        elif self.args.lr_scheduler_type == "cosine":
            lr_scheduler = CosineAnnealingWarmupLR(
                optimizer, total_steps=total_steps, warmup_steps=warmup_steps, eta_min=0.1 * self.args.learning_rate
            )
        else:
            raise NotImplementedError(f"Scheduler type {self.args.lr_scheduler_type} not implemented!")

        # ==============================
        # Boost model
        # ==============================
        if self.args.mixed_precision == "fp16":
            default_dtype = torch.float16
        elif self.args.mixed_precision == "bf16":
            default_dtype = torch.bfloat16
        else:
            default_dtype = torch.float32
        torch.set_default_dtype(default_dtype)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        model, optimizer, _, dataloader, lr_scheduler = self.booster.boost(
            model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
        )
        torch.set_default_dtype(torch.float)

        self.coordinator.print_on_master(f"Model params: {format_numel_str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))}")

        use_pipeline = isinstance(self.booster.plugin, HybridParallelPlugin) and self.booster.plugin.pp_size > 1
        is_pp_last_stage = use_pipeline and self.booster.plugin.stage_manager.is_last_stage()
        print_flag = (not use_pipeline and self.coordinator.is_master()) or (use_pipeline and is_pp_last_stage)

        # ==============================
        # Resume if specified
        # ==============================
        num_steps_per_epoch = len(dataloader)

        if self.args.resume_from_checkpoint is not None:
            self.coordinator.print_on_master(f"Resume training from {self.args.resume_from_checkpoint}...")
            self.load_for_resume(model, optimizer, lr_scheduler, self.args.resume_from_checkpoint)
            self.coordinator.print_on_master(f"Loaded checkpoint {self.args.resume_from_checkpoint} at epoch {self.state.start_epoch} step {self.state.start_step}")

        # if resume training, set the sampler start index to the correct value
        dataloader.sampler.set_start_index(self.state.sample_start_index)

        # ==============================
        # Train
        # ==============================
        for epoch in range(self.state.start_epoch, self.args.num_train_epochs):
            dataloader.sampler.set_epoch(epoch)
            step_nums = num_steps_per_epoch - self.state.start_step
            dataloader_iter = iter(dataloader)

            with tqdm(
                range(step_nums),
                desc=f"Epoch {epoch}",
                disable=not print_flag,
                total=num_steps_per_epoch,
                initial=self.state.start_step,
            ) as pbar:
                for step in pbar:
                    if use_pipeline:
                        outputs = self.booster.execute_pipeline(
                            dataloader_iter, model, lambda outputs, inputs: outputs.loss, optimizer, return_loss=True, return_outputs=True
                        )
                        loss = outputs["loss"]
                    else:
                        batch = next(dataloader_iter)
                        outputs = self.compute_loss(model, batch)
                        loss = outputs[0]
                        self.booster.backward(loss, optimizer)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if not use_pipeline:
                        # reduce and mean
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss.data
                        loss.div_(dist.get_world_size())
                    if print_flag:
                        pbar.set_postfix({"loss": loss.item()})

                    if self.args.logging_steps == "steps" and (self.state.global_step + 1) % self.args.logging_steps == 0:
                        self.coordinator.print_on_master(f"Loss: {loss}")
                        self.state.append_loss(loss.tolist())

                    if self.args.evaluation_strategy == "steps" and (self.state.global_step + 1) % self.args.eval_steps == 0:
                        self.coordinator.print_on_master(f"Evaluating at epoch {epoch} step {step + 1}...")
                        metrics = self.evaluate()
                        self.state.append_metrics(metrics)
                    else:
                        metrics = None

                    if self.args.save_strategy == "steps" and (self.state.global_step + 1) % self.args.save_steps == 0:
                        self.coordinator.print_on_master(f"Saving checkpoint at epoch {epoch} step {step + 1}...")
                        self.save(model, optimizer, lr_scheduler, epoch, step + 1, metrics=metrics)
                    
                    self.state.global_step += 1
                
                if self.args.evaluation_strategy == "epoch":
                    self.coordinator.print_on_master(f"Evaluating at epoch {epoch}...")
                    metrics = self.evaluate()
                    self.state.append_metrics(metrics)
                else:
                    metrics = None

                if self.args.save_strategy == "epoch":
                    self.coordinator.print_on_master(f"Saving checkpoint at epoch {epoch}...")
                    self.save(model, optimizer, lr_scheduler, epoch, step + 1, metrics=metrics)

            # the continue epochs are not resumed, so we need to reset the sampler start index and start step
            dataloader.sampler.set_start_index(0)
            self.state.start_step = 0
    
    def evaluate(self):
        raise NotImplementedError
    
    def get_train_dataloader(
        self,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> DataLoader:
        r"""
        Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.

            :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
        """
        process_group = process_group or dist.distributed_c10d._get_default_group()
        sampler = StatefulDistributedSampler(self.train_dataset, num_replicas=process_group.size(), rank=process_group.rank(), shuffle=shuffle)

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = self.args.seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator
        )

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
        epoch: int,
        step: int,
        metrics: Dict=None
    ):
        """Save model checkpoint and remove worse checkpoints."""
        save_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch}_step{step}")
    
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
            
            if self.coordinator.is_master():
                for checkpoint_to_remove in checkpoints_to_remove:
                    shutil.rmtree(checkpoint_to_remove[0])
            
            self._best_checkpoints.append((save_dir, metrics[self.args.metric_for_best_model]))

        self.coordinator.block_all()

        if self.coordinator.is_master():
            os.makedirs(os.path.join(save_dir), exist_ok=True)

        self.booster.save_model(model, save_dir, shard=True)
        if self.args.save_optimizer:
            self.booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
            self.booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
        if self.coordinator.is_master():
            # NOTE: also save tokenizer
            self.tokenizer.save_pretrained(save_dir)
            self.state.save_to_json(os.path.join(save_dir, "states.json"))

    def load_for_resume(
        self, model: nn.Module, optimizer, lr_scheduler, load_dir
    ) -> Tuple[int, int, int]:
        """Load model checkpoint and optimizer."""
        self.booster.load_model(model, os.path.join(load_dir, "model"))
        self.booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
        self.booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
        self.state = TrainerState.load_from_json(os.path.join(load_dir, "states.json"))

    

def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index



@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0

    start_epoch: int = 0
    start_step: int = 0
    sample_start_index: int = 0

    log_history: List[Dict] = field(default_factory=lambda:[])

    def append_loss(self, loss: float):
        self.log_history.append({"global_step": self.global_step, "loss": loss})

    def append_metrics(self, metrics: Dict):
        self.log_history.append(metrics)
    
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



class ColossalFoldLlamaTrainer(ColossalTrainer):
    def __init__(self, *args, model_args, file_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.file_logger = file_logger

    def compute_loss(self, model, inputs):
        # NOTE: reset memory for each individual input
        if self.model.memory is not None:
            self.model.memory.reset()
        return super().compute_loss(model, inputs)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        from .metrics import Metric
        from .foldllama import evaluate_generation, evaluate_perplexity

        if self.eval_dataset is None:
            return

        dataloader = self.get_eval_dataloader()
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

        # log to file
        if self.args.process_index == 0:
            self.file_logger.log(
                metrics=metrics,
                Model_Args=asdict(self.model_args),
                Training_Args=asdict(self.args),
                Global_Steps=self.state.global_step
            )

        return metrics
