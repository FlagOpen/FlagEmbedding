import os
import torch
import logging
import torch.distributed as dist
from tqdm import tqdm
from dataclasses import asdict
from typing import Optional, List, Dict
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from .metrics import RetrievalMetric
from ..utils.util import save_json
from transformers.trainer_utils import EvalLoopOutput
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(self, *args, corpus:Dataset, model_args, file_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus = corpus
        # handle save/load index/encoding/results
        self.model_args = model_args
        self.file_logger = file_logger
        

    """Trainer with retrieval-based evaluation."""
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        all_args = {
            "model_args": asdict(self.model_args),
            "training_args": asdict(self.args),
        }
        # Good practice: save your training arguments together with the trained model
        save_json(all_args, os.path.join(output_dir, "args.json"))

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return

        args = self.args
        self.model.eval()
        # # make it to fp16
        # dtype = self.model_args.dtype
        # if dtype == "fp16":
        #     dtype = torch.float16
        # else:
        #     dtype = torch.float32
        # self.model.to(dtype)
    
        # NOTE: very important to reset inbatch_same_dataset
        inbatch_same_dataset = self.data_collator.inbatch_same_dataset
        self.data_collator.inbatch_same_dataset = False

        result_path = RetrievalMetric._get_save_path(self.model_args.eval_data, args.output_dir, field="result", save_name=self.model_args.save_name)

        if self.model_args.load_result:
            query_ids, preds, scores = RetrievalMetric._load_result(result_path)

        else:
            if args.eval_method == "retrieval":
                # index corpus
                self.model.index(
                    self.corpus, 
                    output_dir=args.output_dir, 
                    embedding_name=self.model_args.embedding_name,
                    index_factory=self.model_args.faiss_index_factory,
                    load_encode=self.model_args.load_encode,
                    save_encode=self.model_args.save_encode,
                    load_index=self.model_args.load_index, 
                    save_index=self.model_args.save_index,
                    batch_size=self.model_args.batch_size,
                )
                
                # every process uses the same query because the corpus is sharded
                dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=self.model_args.batch_size,
                    pin_memory=True,
                    collate_fn=self.data_collator,
                )

                query_ids = []
                preds = []  # num_samples, hits
                scores = []
                for step, inputs in enumerate(tqdm(dataloader, desc="Searching")):
                    query_id = inputs.pop("query_id")
                    # the indices are already gathered, merged, and sorted inside search function
                    score, indice = self.model.search(inputs["query"], hits=self.model_args.hits)  # batch_size, hits
                    query_ids.extend(query_id.tolist())
                    preds.extend(indice.tolist())
                    scores.extend(score.tolist())

            elif args.eval_method == "rerank":
                dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=self.model_args.batch_size,
                    pin_memory=True,
                    collate_fn=self.data_collator,
                )
                dataloader = self.accelerator.prepare(dataloader)

                query_ids = []
                preds = []  # num_samples, hits
                scores = []
                for step, inputs in enumerate(tqdm(dataloader, desc="Ranking")):
                    inputs = self._prepare_inputs(inputs)
                    query_id = inputs.pop("query_id")
                    key_index = inputs.pop("key_index")         # batch_size, key_num

                    score, indice = self.model.rerank(**inputs, hits=self.model_args.hits) # batch_size, hits

                    # NOTE: when the indices of the keys (w.r.t. the corpus) are provided, we should rerank these indices instead of returning the raw indices
                    # NOTE: when using gather, the index must bigger than -1!
                    gather_index = indice.clone()
                    gather_index[indice == -1] = 0
                    new_indice = key_index.gather(index=gather_index, dim=-1)
                    # NOTE: mask the padded candidate
                    indice = new_indice.masked_fill(indice == -1, -1)

                    query_id = self.accelerator.gather_for_metrics(query_id)
                    # NOTE: important to pad here for later gathering, because different devices may have different key number
                    # FIXME: dim cannot be -1
                    indice = self.accelerator.pad_across_processes(indice, pad_index=-1, dim=1)
                    score = self.accelerator.pad_across_processes(score, pad_index=torch.finfo(score.dtype).min, dim=1)
                    pred = self.accelerator.gather_for_metrics(indice.contiguous())
                    score = self.accelerator.gather_for_metrics(score.contiguous())

                    query_ids.extend(query_id.tolist())
                    preds.extend(pred.tolist())
                    scores.extend(score.tolist())
                    # if step > 4:
                    #     break

            else:
                raise NotImplementedError(f"Eval method {args.eval_method} not implemented!")
            
            if args.process_index == 0 and self.model_args.save_result:
                RetrievalMetric._save_result(query_ids, preds, result_path, scores=scores)

        if args.process_index == 0:
            metrics = [self.compute_metrics(query_ids, preds, scores=scores)]
        else:
            metrics = [None]
            
        # NOTE: broadcast across devices
        dist.broadcast_object_list(metrics, src=0)
        metrics = metrics[0]
        self.accelerator.wait_for_everyone()
        
        # reset
        self.data_collator.inbatch_same_dataset = inbatch_same_dataset
        # self.model.to(torch.float32)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_") and key != "epoch":
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output = EvalLoopOutput(predictions=preds, metrics=metrics, label_ids=None, num_samples=len(preds))
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # log to file
        if args.process_index == 0:
            self.file_logger.log(
                metrics=metrics,
                Model_Args=asdict(self.model_args),
                Training_Args=asdict(args),
                Global_Steps=self.state.global_step
            )

        return output.metrics


class EarlyExitCallBack(TrainerCallback):
    def __init__(self, early_exit_steps=None):
        self.early_exit_steps = early_exit_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.early_exit_steps is not None and state.global_step > self.early_exit_steps:
            control.should_training_stop = True
