import os
import datasets
import time
import torch
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, split_file_dir_name_ext, evaluate_perplexity


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:lm/pg19.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/lm/",
        metadata={'help': 'Output directory for results and logs.'}
    )

    retokenize: bool = field(
        default=False,
        metadata={'help': 'Retokenize the corpus?'}
    )

    padding_side: str = field(
        default="right",
        metadata={'help': 'Which side to pad?'}
    )
    stride: int = field(
        default=2048,
        metadata={'help': 'Streaming stride when evaluating perplexity.'}
    )

    max_sample_num: int = field(
        default=100,
        metadata={'help': 'How many samples to evaluate in eval_data?'}
    )
    min_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Minimum length for input_ids.'}
    )


def process_lm_pre(tokenizer, tokenize_max_char=None):
    def _process(data):
        outputs = {'input_ids': []}
        for text in data['text']:
            if tokenize_max_char is not None:
                text = text[:tokenize_max_char]

            outputs['input_ids'].append(tokenizer.encode(text, add_special_tokens=False))
        return outputs
    return _process


def process_lm(tokenizer, max_length=4096, stride=1024, min_length=None):
    # stride=0 indicates we just use one forward pass with max_length for each text
    if stride == 0:
        stride = max_length
        jump = True
    else:
        jump = False

    test = tokenizer.encode("test")
    has_bos = False
    if test[0] == tokenizer.bos_token_id:
        # NOTE: subtract 1 because it will be occupied by the bos token
        max_length -= 1
        has_bos = True

    def _process(data, indices, **kwds):
        outputs = defaultdict(list)

        for text, index in zip(data["text"], indices):
            input_ids = tokenizer.encode(text, add_special_tokens=False)

            seq_len = len(input_ids)
            prev_end_loc = 0

            if min_length is not None and seq_len < min_length:
                continue

            for start_loc in range(0, seq_len, stride):
                end_loc = min(start_loc + max_length, seq_len)
                sub_seq_len = end_loc - start_loc
                sub_trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

                sub_input_ids = input_ids[start_loc: end_loc]
                sub_attention_mask = [1 for _ in range(sub_seq_len)]
                if has_bos:
                    sub_input_ids.insert(0, tokenizer.bos_token_id)
                    sub_attention_mask.insert(0, 1)
                    sub_seq_len += 1

                sub_labels = sub_input_ids.copy()
                sub_labels[:-sub_trg_len] = [-100 for _ in range(sub_seq_len - sub_trg_len)]

                sub_inputs = {
                    "index": index,
                    "input_ids": sub_input_ids,
                    "attention_mask": sub_attention_mask,
                    "labels": sub_labels,
                }

                for k, v in sub_inputs.items():
                    outputs[k].append(v)
                
                prev_end_loc = end_loc
                # NOTE: when end_loc is just the same as seq_len, jump out
                if end_loc == seq_len or jump:
                    break

        return outputs
    return _process


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    # increase timeout to avoid error
    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    _, dataset_name, _ = split_file_dir_name_ext(args.eval_data)

    process_fn = process_lm(tokenizer, max_length=args.max_length, stride=args.stride, min_length=args.min_length)
    dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
    if len(dataset) > args.max_sample_num:
        # slice out the first max_sample_num samples
        dataset = dataset.train_test_split(args.max_sample_num, shuffle=False)["test"]
    dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, keep_in_memory=True, with_indices=True)

    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        # only pin memory when no gpu
        pin_memory=not args.cpu,
    )
    accelerator.wait_for_everyone()

    # NOTE: prepare dataloader so the data moves to GPU automatically
    dataloader = accelerator.prepare(dataloader)

    t1 = time.time()
    perplexity = evaluate_perplexity(model, dataloader, accelerator)
    t2 = time.time()
    memory = torch.cuda.max_memory_allocated() / 1024**2
    metrics = {"perplexity": perplexity, "time": round((t2 - t1) / len(dataset), 4), "memory": memory}

    if accelerator.process_index == 0:
        log_path = os.path.join(args.output_dir, f"{dataset_name}.log")

        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
