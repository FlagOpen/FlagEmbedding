import os
import torch
import time
import datasets
from typing import List, Optional
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from functools import partial

from src.data import Data
from src.metrics import Metric
from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, evaluate_perplexity, split_file_dir_name_ext, apply_chat_template

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: Optional[str] = field(
        default="long-llm:sharegpt/3-turn.json",
        metadata={'help': 'Evaluation json data.'}
    )
    output_dir: str = field(
        default="data/results/multiturn/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )

    min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for evaluation?'}
    )
    # no more than 1536 tokens because gist cannot process more
    max_length: int = field(
        default=100000,
        metadata={'help': 'How many tokens at maximum for evaluation?'}
    )

    num_turn: int = field(
        default=3,
        metadata={'help': 'How many turns?'}
    )
    breakdown: bool = field(
        default=False,
    )


def process_multiturn(data, indices, tokenizer, chat_template, min_length, max_length, num_turn=None, breakdown=False):
    outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

    # accumulative
    if breakdown:
        for i, source in enumerate(data['accum_conversations']):
            # break the multi-turn conversation
            if num_turn is None:
                num_turn = len(source) // 2

            # skip conversations that do not have enough turns
            if num_turn * 2 > len(source):
                continue

            for j in range(0, 2 * num_turn, 2):
                turn_source = source[j: j + 2]
                encoded = apply_chat_template(
                    chat_template, 
                    turn_source, 
                    tokenizer=tokenizer, 
                    return_labels=True,
                ).encoded

                # skip data that not fall in between min_length and max_length
                if min_length is not None and len(encoded["input_ids"]) < min_length:
                    continue
                if max_length is not None and len(encoded["input_ids"]) > max_length:
                    continue

                for k, v in encoded.items():
                    outputs[k].append(v)
                outputs['length'].append(len(encoded['input_ids']))
                # NOTE: the breakdown conversations belong to the same root
                outputs['index'].append(indices[i])

        return outputs
    
    else:
        for i, source in enumerate(data['conversations']):
            if num_turn is not None:
                source = source[:2 * num_turn]

            encoded = apply_chat_template(
                chat_template, 
                source, 
                tokenizer=tokenizer, 
                return_labels=True,
            ).encoded

            # skip data that not fall in between min_length and max_length
            if min_length is not None and len(encoded["input_ids"]) < min_length:
                continue
            if max_length is not None and len(encoded["input_ids"]) > max_length:
                continue

            for k, v in encoded.items():
                outputs[k].append(v)
            outputs['length'].append(len(encoded['input_ids']))
            outputs['index'].append(indices[i])

        return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, split="train", cache_dir=args.dataset_cache_dir)

        process_fn = partial(
            process_multiturn,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            min_length=args.min_length,
            num_turn=args.num_turn,
            breakdown=args.breakdown,
        )
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, batch_size=10, with_indices=True, remove_columns=raw_dataset.column_names)

    # get labels (the target generation result)
    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        # only pin memory when no gpu
        pin_memory=not args.cpu,
    )

    if not args.enable_tp:
        model, dataloader = accelerator.prepare(model, dataloader)
        # NOTE: unwrap because we just use the model for evaluation
        model = accelerator.unwrap_model(model)
    else:
        # NOTE: prepare dataloader so the data moves to GPU automatically
        dataloader = accelerator.prepare(dataloader)

    accelerator.wait_for_everyone()

    accelerator.print(dataset['index'])

    t1 = time.time()
    perplexity = evaluate_perplexity(model, dataloader, accelerator)
    t2 = time.time()

    t = [t2 - t1]
    if accelerator.num_processes > 1:
        t = accelerator.gather_for_metrics(t)
    t = sum(t)
    metrics = {"perplexity": perplexity, "time": round(t, 4)}

    if accelerator.process_index == 0:
        log_path = os.path.join(args.output_dir, f"metrics.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))



if __name__ == "__main__":
    main()
