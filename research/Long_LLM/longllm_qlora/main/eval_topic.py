# modified based on https://github.com/DachengLi1/LongChat/blob/longeval/longeval/eval.py

import os
import json
import torch
import datasets
from tqdm import tqdm
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
    do_sample: bool = False

def process_topic_retrieval(tokenizer, chat_template, num_topic):
    def _process(data):
        outputs = {'input_ids': [], 'attention_mask': [], 'target': [], 'length': [], 'num': []}
        
        for context, question, topics, num in zip(data['context'], data['question'], data['topics'], data['num_topics']):
            # filter out samples that do not have proper number of topics/lines
            if num not in num_topic:
                continue

            prompt = " ".join([context, question])
            # the question always asks for the first topic
            target = topics[0]

            encoded = apply_chat_template(chat_template, [{'role': 'user', 'content': prompt}], tokenizer=tokenizer, add_generation_prompt=True).encoded

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

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        process_fn = process_topic_retrieval(
            tokenizer,
            chat_template=args.chat_template,
            num_topic=args.num_topic,
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, remove_columns=raw_dataset.column_names)
        # group instances of the same number of topics together, so that their lengths are approximately equal
        groupby_dataset = dataset.to_pandas().groupby("num")

    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    
    accuracy = {}
    f1_score = {}
    results = defaultdict(list)

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

        all_lengths = []
        all_outputs = []
        for i, x in enumerate(tqdm(dataloader, desc=f"Evaluating {num} Topics")):
            # NOTE: important to reset memory for every batch
            if hasattr(model, "memory"):
                model.memory.reset()

            length = x.pop("length")

            # accelerator.print(tokenizer.decode(x["input_ids"][0]))

            outputs = model.generate(
                **x, 
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                # FIXME: sometimes transformers cannot detect deepspeed zero3, dont know why
                synced_gpus=accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3,
            )
            start_idx = x["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

            if accelerator.num_processes > 1:
                outputs = accelerator.pad_across_processes(outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
                outputs = accelerator.gather_for_metrics(outputs)
                length = accelerator.gather_for_metrics(length)
            
            outputs = outputs.tolist()
            length = length.tolist()

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(outputs)
            all_lengths.extend(length)

        length = int(sum(all_lengths) / len(all_lengths))

        acc = 0
        f1 = 0
        for output, target in zip(all_outputs, all_targets):
            if target.lower() in output.lower():
                acc += 1
            else:
                acc += 0
            f1 += round(qa_f1_score(output, target), 4)
            results[length].append({"target": target, "prediction": output})

        acc /= len(all_outputs)
        f1 /= len(all_outputs)

        accuracy[length] = acc
        f1_score[length] = f1
    
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
