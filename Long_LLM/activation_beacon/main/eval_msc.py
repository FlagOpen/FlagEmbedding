import os
import json
import torch
import datasets
from rouge import Rouge
from tqdm import tqdm
from typing import List, Optional
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import partial

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, split_file_dir_name_ext, apply_chat_template, normalize_text
from .longbench_utils import qa_f1_score

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:memgpt/msc.json",
        metadata={'help': 'Evaluation json data.'}
    )
    output_dir: str = field(
        default="data/results/msc/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    chat_template: str = field(
        default='no'
    )
    max_length: int = field(
        default=None
    )
    do_sample: bool = False
    max_new_tokens: int = 20



def process_msc(data, tokenizer, max_length, chat_template):
    outputs = {'input_ids': [], 'attention_mask': [], 'target': []}
    
    for context, input_, output in zip(data['context'], data['input'], data['output']):
        prompt = context + "\n" + input_

        if max_length is not None:
            prompt = tokenizer.decode(tokenizer.encode(prompt, add_special_tokens=False)[-max_length:])

        encoded = apply_chat_template(chat_template, [{'role': 'user', 'content': prompt}], tokenizer=tokenizer, add_generation_prompt=True).encoded
        encoded["target"] = output
    
        for k, v in encoded.items():
            outputs[k].append(v)
    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        process_fn = partial(process_msc, tokenizer=tokenizer, chat_template=args.chat_template, max_length=args.max_length)
        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, remove_columns=raw_dataset.column_names)

    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    
    results = []

    all_targets = dataset["target"]
    dataset = dataset.remove_columns(["target"])
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        # only pin memory when no gpu
        pin_memory=not args.cpu,
    )

    if not args.enable_tp:
        # NOTE: prepare model only once
        if len(accelerator._models) == 0:
            model, dataloader = accelerator.prepare(model, dataloader)
            model = accelerator.unwrap_model(model)
        else:
            dataloader = accelerator.prepare(dataloader)
    else:
        # NOTE: prepare dataloader so the data moves to GPU automatically
        dataloader = accelerator.prepare(dataloader)

    all_outputs = []
    for i, x in enumerate(tqdm(dataloader)):
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        output = model.generate(**x)

        if isinstance(output, torch.Tensor):
            # 1, max_new_tokens
            output = output[:, x['input_ids'].shape[1]:]
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
        elif isinstance(output, list):
            pass

        if accelerator.num_processes > 1:
            output = accelerator.gather_for_metrics(output)

        all_outputs.extend(output)

    if accelerator.process_index == 0:
        rouge = Rouge()
        score = rouge.get_scores(normalize_text(all_outputs), normalize_text(all_targets), avg=True)["rouge-l"]["r"]

        for output, target in zip(all_outputs, all_targets):
            results.append({"target": target, "prediction": output})

        result_dir = os.path.join(args.output_dir, args.result_dir) if args.result_dir is not None else args.output_dir
        with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
            json.dump(results, f)
        # also save config
        args.save(os.path.join(result_dir, "config.json"))

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log({'rouge': score}, Args=asdict(args))


if __name__ == "__main__":
    main()
