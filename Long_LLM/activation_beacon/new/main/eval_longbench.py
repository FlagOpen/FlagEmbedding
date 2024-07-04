import os
import datasets
import json
import torch
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template
from .longbench_utils import DATASET2PROMPT, DATASET2MAXNEWTOKENS, DATASET2CATEGORY, scorer

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:longbench/",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/longbench/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    tasks: List[str] = field(
        default_factory=lambda: ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum', 'lcc', 'repobench-p'],
        metadata={'help': 'Which dataset to evaluate?'}
    )
    newline_as_eos: bool = field(
        default=True,
        metadata={'help': 'Whether to use new line as eos (for QA tasks only) or not.'}
    )

    max_length: int = field(
        default=31500,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )

    do_sample: bool = False


def process_longbench(data, indices, tokenizer, chat_template, task, max_length=3500, truncate_from_middle=True):
    outputs = {'input_ids': [], 'attention_mask': [], "index": []}

    for input, context, index in zip(data['input'], data['context'], indices):
        prompt_template = DATASET2PROMPT[task]
        prompt = prompt_template.format(input=input, context=context)

        if truncate_from_middle:
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            tokenized_prompt = tokenizer.encode(prompt)
            prompt = tokenizer.decode(tokenized_prompt[-max_length:], skip_special_tokens=True)

        # in fewshot learning and code completion we do not need chat template
        if not any(x in DATASET2CATEGORY[task] for x in ["Few-Shot Learning", "Code Completion"]):
            encoded = apply_chat_template(
                chat_template, 
                messages=[{'role': 'user', 'content': prompt}],
                tokenizer=tokenizer,
                add_generation_prompt=True,
            ).encoded
        else:
            encoded = tokenizer(prompt)

        outputs["input_ids"].append(encoded["input_ids"])
        outputs["attention_mask"].append(encoded["attention_mask"])
        outputs["index"].append(index)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    if hasattr(model, "generation_config"):
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    # stop generation for QA tasks when \n appears
    if args.newline_as_eos:
        eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

    if args.tasks == ["all"]:
        tasks = list(DATASET2PROMPT.keys())
    else:
        tasks = args.tasks

    with accelerator.main_process_first():
        all_datasets = {}

        for task in tasks:
            process_fn = partial(
                process_longbench,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                task=task,
                max_length=args.max_length,
                truncate_from_middle=args.truncate_from_middle,
            )

            path = os.path.join(args.eval_data, f"{task}.jsonl")
            raw_dataset = datasets.load_dataset("json", data_files=path, cache_dir=args.dataset_cache_dir, split="train")
            dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, batch_size=10, with_indices=True, remove_columns=raw_dataset.column_names)

            all_datasets[task] = (raw_dataset, dataset)

    result_dir = os.path.join(args.output_dir, args.result_dir)

    metrics = {}

    for i, task in enumerate(all_datasets.keys()):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {task} ({i + 1} / {len(all_datasets)})...")

        result_path = os.path.join(result_dir, f"{task}.json")

        raw_dataset, dataset = all_datasets[task]

        if not (args.load_result and os.path.exists(result_path)):
            data_collator = DefaultDataCollator(tokenizer=tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                collate_fn=data_collator,
                # only pin memory when no gpu
                pin_memory=not args.cpu,
            )

            dataloader = accelerator.prepare(dataloader)

            indices = []
            preds = []
            max_new_tokens = DATASET2MAXNEWTOKENS[task]

            for i, x in enumerate(tqdm(dataloader, desc="Generating")):
                index = x.pop("index").tolist()
                input_length = x["input_ids"].shape[1]

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory"):
                    model.memory.reset()

                kwargs = {"max_new_tokens": max_new_tokens}
                if task in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
                    kwargs["eos_token_id"] = eos_token_id

                # NOTE: very important to include \n as an eos token for QA tasks, otherwise the F1 score is devastating
                output = model.generate(
                    **x,
                    **kwargs
                )
                if isinstance(output, torch.Tensor):
                    # 1, max_new_tokens
                    output = output[:, input_length:]
                    output = tokenizer.batch_decode(output, skip_special_tokens=True)
                elif isinstance(output, list):
                    pass

                if accelerator.num_processes > 1:
                    output = accelerator.gather_for_metrics(output)
                    index = accelerator.gather_for_metrics(index)

                if accelerator.process_index == 0:
                    preds.extend(output)
                    indices.extend(index)
        else:
            if accelerator.process_index == 0:
                preds = []
                indices = []

                with open(result_path, "r", encoding="utf-8") as f:
                    # the first line is the metric score
                    f.readline()

                    for line in f:
                        item = json.loads(line)
                        preds.append(item["pred"])
                        indices.append(len(indices))

        if accelerator.process_index == 0:
            answers = raw_dataset["answers"]
            lengths = raw_dataset["length"]
            all_classes = raw_dataset["all_classes"][0]
            score = scorer(task, preds, answers, all_classes)

            logger.info(f"{task}: {score}")
            metrics[task] = score

            with open(makedirs(result_path), "w", encoding="utf-8") as f:
                f.write(json.dumps(score, ensure_ascii=False) + "\n")
                for index, pred in zip(indices, preds):
                    sample = raw_dataset[index]
                    del sample["all_classes"]
                    del sample["context"]
                    del sample["language"]
                    del sample["_id"]
                    sample["pred"] = pred
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))

        # compute category score
        category_metrics = defaultdict(list)
        for dataset, metric in metrics.items():
            category = DATASET2CATEGORY[dataset]
            category_metrics[category].append(metric)
        for k, v in category_metrics.items():
            # when evaluating on longbench_e, each metric is a dict of float
            if isinstance(v[0], dict):
                category_metric = {}
                for kk in v[0].keys():
                    vv = [v[j][kk] for j in range(len(v))]
                    category_metric[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = category_metric
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)
        
        # compute average score
        if isinstance(next(iter(metrics.values())), dict):
            avg = defaultdict(list)
            for k, v in metrics.items():
                for kk, vv in v.items():
                    avg[kk].append(vv)
            for k, v in avg.items():
                avg[k] = round(sum(v) / len(v), 2)
        else:
            avg = round(sum(metrics.values()) / len(metrics), 2)
        metrics["avg"] = avg

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args), Category_Metrics=category_metrics)


if __name__ == "__main__":
    main()
