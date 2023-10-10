import os
import logging
import datasets
import torch
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict

from src.lm import SRLMArgs, SelfRetrievalLM
from src.retrieval import Retriever, RetrievalArgs, TASK_CONFIG
from src.utils.util import makedirs, pad_nested_lists, get_max_length_in_nested_lists, FileLogger

logger = logging.getLogger(__name__)
import transformers
# disable too long input warning
transformers.logging.set_verbosity_error()


# merge two args to get unified arguments
@dataclass
class LRLMArgs(RetrievalArgs, SRLMArgs):
    eval_data: str = field(
        default="llm-embedder:chat/msc/test.json",
        metadata={'help': 'Evaluation file containing long texts.'}
    )
    lm_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    add_position_ids: bool = field(
        default=False,
        metadata={'help': 'Create position ids based on attention masks? Useful when training left-padded models with absolute position embeddings.'}
    )
    key_num: int = field(
        default=1,
        metadata={'help': 'How many chunks to retrieve at a time?'}
    )
    log_path: str = field(
        default="data/results/msc/msc.log",
        metadata={'help': 'Path to the file for logging.'}
    )
    debug_retrieval: bool = field(
        default=False,
        metadata={'help': 'Check retrieval queries and values?'}
    )


@dataclass
class HistoryCollator:
    """Collate histories, pad them, and return masks"""
    def __call__(self, batch_elem):
        first_elem = batch_elem[0]
        return_batch = {}

        for key, value in first_elem.items():
            batch_value = [elem[key] for elem in batch_elem]
            if key == "history":
                longest = get_max_length_in_nested_lists(batch_value)
                batch_value, history_mask = pad_nested_lists(batch_value, longest, "", "right")
                history_mask = torch.tensor(history_mask, dtype=torch.bool)
                return_batch["history_mask"] = history_mask

            elif key == "answers":
                # there is only one answer
                key = "answer"
                batch_value = [elem[0] for elem in batch_value]
            
            elif key in ["query_id", "task"]:
                continue

            # strip here for convenience
            return_batch[key] = np.char.strip(np.array(batch_value))
        return return_batch


def main():
    parser = HfArgumentParser([LRLMArgs])
    args, = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(cpu=args.cpu)

    retriever = Retriever(
        retrieval_method=args.retrieval_method,
        # for dense retriever
        query_encoder=args.query_encoder,
        key_encoder=args.key_encoder,
        pooling_method=args.pooling_method,
        dense_metric=args.dense_metric,
        query_max_length=args.query_max_length,
        key_max_length=args.key_max_length,
        tie_encoders=args.tie_encoders,
        truncation_side=args.truncation_side,
        cache_dir=args.model_cache_dir, 
        dtype=args.dtype,
        accelerator=accelerator,
        # for bm25 retriever
        anserini_dir=args.anserini_dir,
        k1=args.k1,
        b=args.b
    )

    if args.add_instruction:
        instruction = TASK_CONFIG[args.version]["instruction"]["chat"]
    else:
        instruction = None

    lm = SelfRetrievalLM(
        model_name_or_path=args.model_name_or_path,
        retriever=retriever,
        dtype=args.lm_dtype,
        device_map=args.lm_device_map,
        padding_side=args.padding_side,
        cache_dir=args.model_cache_dir,
        context_window_size=args.context_window_size,
        chunk_size=args.chunk_size,
        key_num=args.key_num,
        chunk_batch_size=args.chunk_batch_size,
        retrieval_method=args.retrieval_method,
        order_method=args.order_method,
        integrate_method=args.integrate_method,
        instruction=instruction,
        debug_retrieval=args.debug_retrieval,
        add_sep=args.add_sep,
        accelerator=accelerator,
    )

    logging.info(f"Loading data from {args.eval_data}...")

    with accelerator.main_process_first():
        dataset = datasets.load_dataset("json", data_files=args.eval_data, split="train", cache_dir=args.dataset_cache_dir)

    data_collator = HistoryCollator()
    dataloader = DataLoader(
        dataset, 
        batch_size=args.lm_batch_size, 
        collate_fn=data_collator,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    perplexity = lm.compute_perplexity(dataloader)
    metrics = {"perplexity": perplexity}

    if accelerator.process_index == 0:
        log_path = os.path.join(args.log_path)

        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
