import os
import torch
import datasets
import numpy as np
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset
from transformers.trainer import Trainer, PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME, has_length, is_datasets_available, RandomSampler
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from transformers.utils import logging

logger = logging.get_logger(__name__)


class FoldLlamaTrainer(Trainer):
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
        self.model.memory.reset()
        return inputs
    
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # NOTE: do not save optimizer
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        # NOTE: disable saving optimizer state from DeepSpeed
        # if self.is_deepspeed_enabled:
        #     # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        #     # config `stage3_gather_16bit_weights_on_model_save` is True
        #     self.model_wrapped.save_checkpoint(output_dir)
        
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        
        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        from .activation_beacon_llama import evaluate_generation, evaluate_perplexity
        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return

        dataloader = self.get_eval_dataloader()
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model
            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

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
        else:
            raise NotImplementedError(f"Eval method {self.args.eval_method} not implemented!")

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


class CurriculumFoldCallBack(TrainerCallback):
    """Callback for curriculum learning of activation fold."""
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.curriculum_steps == 0:
            index = state.global_step // args.curriculum_steps
            if len(args.curriculum_fold_window_and_sizes) > 1:
                fold_window_and_sizes = args.curriculum_fold_window_and_sizes[index]
            else:
                fold_window_and_sizes = args.curriculum_fold_window_and_sizes[0]
            if len(args.curriculum_fold_mix_method) > 1:
                fold_mix_method = args.curriculum_fold_mix_method[index]
            else:
                fold_mix_method = args.curriculum_fold_mix_method[0]
            model = kwargs["model"]
            model.config.fold_window_and_sizes = fold_window_and_sizes
            model.config.fold_mix_method = fold_mix_method
            # NOTE: reset memory with new fold_window_and_sizes
            model.set_memory()
