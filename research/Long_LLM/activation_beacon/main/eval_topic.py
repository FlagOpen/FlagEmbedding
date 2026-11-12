# modified based on https://github.com/DachengLi1/LongChat/blob/longeval/longeval/eval.py

import os
import json
import torch
import datasets
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import List, Optional
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, split_file_dir_name_ext, apply_chat_template
from .longbench_utils import qa_f1_score

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:longeval/topic_retrieval.json",
        metadata={'help': 'Evaluation json data.'}
    )
    output_dir: str = field(
        default="data/results/topic_retrieval/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )
    num_topic: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30, 40, 50, 60, 70],
        metadata={'help': 'How many topics to in the conversation?'}
    )
    adapt_window: bool = field(
        default=False,
        metadata={'help': 'Dynamically change the beacon window so that the input is always compressed?'}
    )
    target_topic: str = field(
        default="first",
        metadata={'help': 'Which topic to evaluate?'}
    )

    do_sample: bool = False
    max_new_tokens: int = 50

def process_topic_retrieval(data, tokenizer, chat_template, num_topic, target_topic):
    outputs = {'input_ids': [], 'attention_mask': [], 'target': [], 'length': [], 'num': []}
    
    for context, question, topics, num in zip(data['context'], data['question'], data['topics'], data['num_topics']):
        # filter out samples that do not have proper number of topics/lines
        if num not in num_topic:
            continue

        if num == 1:
            context = context.split(" \n USER: Great, this is the end of our discussion")[0]
            context = context + " Now the record ends."

        if target_topic == "first":
            question = f"What is the first topic we have discussed? Only give me the topic name. Do not summarize yourself."
            target = topics[0]
        elif target_topic == "random":
            target_idx = np.random.randint(0, num)
            question = f"What is the No.{target_idx} topic we have discussed? Only give me the topic name. Do not summarize yourself."
            target = topics[target_idx]
        else:
            raise NotImplementedError

        prompt = " ".join([context, question])
        # the question always asks for the first topic

        encoded = apply_chat_template(chat_template, [{'role': 'user', 'content': prompt}], tokenizer=tokenizer, add_generation_prompt=True).encoded

        encoded["target"] = target
        encoded["length"] = len(encoded.input_ids)
        encoded["num"] = num

        for k, v in encoded.items():
            if k in outputs:
                outputs[k].append(v)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        process_fn = partial(process_topic_retrieval,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            num_topic=args.num_topic,
            target_topic=args.target_topic,
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, remove_columns=raw_dataset.column_names)
        # group instances of the same number of topics together, so that their lengths are approximately equal
        groupby_dataset = dataset.to_pandas().groupby("num")

    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    
    accuracy = {}
    f1_score = {}
    results = defaultdict(list)
    # used for adapt_window
    if args.adapt_window:
        beacon_window = getattr(model.config, "beacon_window", None)

    for num, dataset in groupby_dataset:
        dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(num), preserve_index=False)
        all_targets = dataset["target"]
        # remove unnecessary columns
        dataset = dataset.remove_columns(["target", "num"])

        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            collate_fn=data_collator,
            # only pin memory when no gpu
            pin_memory=not args.cpu,
        )

        # NOTE: prepare dataloader so the data moves to GPU automatically
        dataloader = accelerator.prepare(dataloader)

        all_lengths = []
        all_outputs = []
        for i, x in enumerate(tqdm(dataloader, desc=f"Evaluating {num} Topics")):
            # NOTE: important to reset memory for every batch
            if hasattr(model, "memory"):
                if args.adapt_window:
                    length = x['length'][0].item()
                    if length < beacon_window:
                        beacon_window = (length // 256) * 256
                        beacon_stride = beacon_window
                        model.memory.set(
                            beacon_window=beacon_window,
                            beacon_stride=beacon_stride,
                        )

                model.memory.reset()

            length = x.pop("length").tolist()

            output = model.generate(**x)

            if isinstance(output, torch.Tensor):
                # 1, max_new_tokens
                output = output[:, x['input_ids'].shape[1]:]
                output = tokenizer.batch_decode(output, skip_special_tokens=True)
            elif isinstance(output, list):
                pass

            if accelerator.num_processes > 1:
                output = accelerator.gather_for_metrics(output)
                length = accelerator.gather_for_metrics(length)
            
            all_outputs.extend(output)
            all_lengths.extend(length)

        length = int(sum(all_lengths) / len(all_lengths))

        acc = 0
        f1 = 0
        for output, target in zip(all_outputs, all_targets):
            if target.lower() in output.lower():
                acc += 1
            else:
                acc += 0
            f1 += qa_f1_score(output, target)
            results[length].append({"target": target, "prediction": output})

        acc /= len(all_outputs)
        f1 /= len(all_outputs)

        accuracy[length] = acc
        f1_score[length] = round(f1, 4)
    
    if accelerator.process_index == 0:
        result_dir = os.path.join(args.output_dir, args.result_dir) if args.result_dir is not None else args.output_dir
        with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
            json.dump(results, f)
        # also save config
        args.save(os.path.join(result_dir, "config.json"))

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log({'accuracy': accuracy, 'f1': f1_score}, Args=asdict(args))


if __name__ == "__main__":
    main()
