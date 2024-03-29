# modified based on https://github.com/DachengLi1/LongChat/blob/longeval/longeval/eval.py

import os
import torch
import datasets
from tqdm import tqdm
from typing import List
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, split_file_dir_name_ext, apply_chat_template
from .longbench_utils import qa_f1_score

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="activation-beacon:longeval/topic_retrieval.json",
        metadata={'help': 'Evaluation json data.'}
    )
    output_dir: str = field(
        default="data/results/longeval/",
        metadata={'help': 'Output directory for results and logs.'}
    )
    topic_num: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 25],
        metadata={'help': 'How many topics to in the conversation?'}
    )
    line_num: List[int] = field(
        default_factory=lambda: [200, 300, 400, 500, 600, 680],
        metadata={'help': 'How many lines to in the conversation?'}
    )


def process_longeval(tokenizer, chat_template, data_type, topic_num, line_num):
    if data_type == "topic":
        target_nums = set(topic_num)
    elif data_type == "line":
        target_nums = set(line_num)
    else:
        raise ValueError(f"Invalid data type {data_type}!")
    
    def _process(data):
        outputs = {'input_ids': [], 'attention_mask': [], 'target': [], 'length': [], 'num': []}
        
        if data_type == "topic":
            nums = data['num_topics']
        else:
            nums = data['num_lines']

        for prompt, target, num in zip(data['prompt'], data['target'], nums):
            if num not in target_nums:
                continue

            # filter out samples that do not have proper number of topics/lines
            if chat_template != 'no':
                prompt = apply_chat_template(chat_template, [{'role': 'user', 'content': prompt}], add_generation_prompt=True)

            encoded = tokenizer(prompt)
            encoded["target"] = target
            encoded["length"] = len(encoded.input_ids)
            encoded["num"] = num

            for k, v in encoded.items():
                outputs[k].append(v)

        return outputs
    return _process


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    # topic or line
    data_type = split_file_dir_name_ext(args.eval_data)[1].split("_")[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)

    with accelerator.main_process_first():
        process_fn = process_longeval(
            tokenizer,
            chat_template=args.chat_template,
            data_type=data_type,
            topic_num=args.topic_num,
            line_num=args.line_num
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, remove_columns=raw_dataset.column_names)
        groupby_dataset = dataset.to_pandas().groupby("num")

    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    metrics = {}

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
        # shard dataloader
        dataloader = accelerator.prepare(dataloader)

        all_lengths = []
        all_outputs = []
        for i, x in enumerate(tqdm(dataloader, desc=f"Evaluating {num} {data_type}")):
            # NOTE: important to reset memory for every batch
            if hasattr(model, "memory") and model.memory is not None:
                model.memory.reset()

            length = x.pop("length")

            outputs = model.generate(
                **x, 
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
            )
            start_idx = x["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

            # must be contiguous
            outputs = outputs.contiguous()
            outputs = accelerator.pad_across_processes(outputs, pad_index=tokenizer.pad_token_id, dim=1)
            outputs = accelerator.gather_for_metrics(outputs).tolist()
            length = accelerator.gather_for_metrics(length).tolist()

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(outputs)
            all_lengths.extend(length)

            accelerator.print(f"Target:{all_targets[i]}\nPred:{repr(all_outputs[i])}")
    
        accuracy = 0
        f1 = 0
        for output, target in zip(all_outputs, all_targets):
            if target.lower() in output.lower():
                accuracy += 1
            else:
                accuracy += 0
            f1 += round(qa_f1_score(output, target), 4)
        accuracy /= len(all_outputs)
        f1 /= len(all_outputs)
        length = int(sum(all_lengths) / len(all_lengths))
        metrics[length] = {
            "accuracy": accuracy,
            "f1": f1,
        }
    
    if accelerator.process_index == 0:
        log_path = os.path.join(args.output_dir, f"{data_type}_retrieval.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
