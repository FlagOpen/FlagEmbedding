import os
import datasets
import json
import torch
import pandas as pd
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser, AutoTokenizer
from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template
from .infbench_utils import TASK_TO_PATH, TASK_TO_MAX_NEW_TOKENS, get_score_one, create_prompt, get_answer


logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:infbench",
        metadata={'help': 'The directory of all infbench evaluation data.'}
    )
    output_dir: str = field(
        default="data/results/infbench/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    tasks: List[str] = field(
        default_factory=lambda: ['longbook_qa_eng'],
        metadata={'help': 'Which dataset to evaluate?'}
    )
    prompt_template: str = field(
        default="mistral",
        metadata={'help': 'Which prompt template to use? (See infbench_utils.py for reference.)'}
    )

    max_length: int = field(
        default=100000,
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


def process_infbench(data, indices, tokenizer, chat_template, task:str, prompt_template:str="mistral", max_length=100000, truncate_from_middle=True):
    outputs = {'input_ids': [], 'attention_mask': [], "index": [], "answer": []}

    # NOTE: high version datasets use LazyBatch to wrap data, which cannot be reverted to list of dicts, thus, we need to convert it to dict first
    data = pd.DataFrame(dict(data)).to_dict(orient="records")

    for sample, index in zip(data, indices):
        prompt = create_prompt(sample, task, prompt_template)
        answer = get_answer(sample, task)

        if truncate_from_middle:
            tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            prompt = tokenizer.decode(tokenized_prompt[-max_length:], skip_special_tokens=True)

        encoded = apply_chat_template(
            chat_template,
            messages=[{'role': 'user', 'content': prompt}],
            tokenizer=tokenizer,
            add_generation_prompt=True,
        ).encoded

        for k, v in encoded.items():
            outputs[k].append(v)

        outputs["index"].append(index)
        outputs["answer"].append(answer)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        all_datasets = {}

        for task in args.tasks:
            process_fn = partial(
                process_infbench, 
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                task=task,
                prompt_template=args.prompt_template,
                truncate_from_middle=args.truncate_from_middle,
            )

            path = os.path.join(args.eval_data, TASK_TO_PATH[task])
            raw_dataset = datasets.load_dataset("json", data_files=path, cache_dir=args.dataset_cache_dir, split="train")
            dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, batch_size=10, with_indices=True, remove_columns=raw_dataset.column_names)

            all_datasets[task] = dataset

    result_dir = os.path.join(args.output_dir, args.result_dir)

    metrics = {}

    for i, (task, dataset) in enumerate(all_datasets.items()):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {task} ({i + 1} / {len(all_datasets)})...")

        result_path = os.path.join(result_dir, f"{task}.json")
        
        if args.load_result and os.path.exists(result_path):
            if accelerator.process_index == 0:
                scores = []
                preds = []
                labels = []
                indices = []
                with open(result_path, encoding="utf-8") as f:
                    # the first line is metric
                    f.readline()

                    for line in f:
                        item = json.loads(line)
                        pred = item["pred"]
                        label = item["label"]
                        index = item["index"]
                        # NOTE: here we explicitly input model_name=None
                        score = get_score_one(pred, label, task, None)
                        scores.append(score)

                        preds.append(pred)
                        labels.append(label)
                        indices.append(index)

                    score = round(sum(scores) / len(scores), 4)

                    logger.info(f"{task}: {score}")
                    metrics[task] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for index, pred, label in zip(indices, preds, labels):
                        item = {
                            "index": index,
                            "pred": pred,
                            "label": label,
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

        else:
            # get answers in advance
            labels = dataset["answer"]
            dataset = dataset.remove_columns(["answer"])

            data_collator = DefaultDataCollator(tokenizer=tokenizer)
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

            indices = []
            preds = []
            max_new_tokens = TASK_TO_MAX_NEW_TOKENS[task]

            for j, x in enumerate(tqdm(dataloader, desc="Generating")):
                index = x.pop("index")[0]
                input_length = x["input_ids"].shape[1]

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory"):
                    model.memory.reset()

                output = model.generate(
                    **x,
                    max_new_tokens=max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    # FIXME: sometimes transformers cannot detect deepspeed zero3, dont know why
                    synced_gpus=accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3,
                )

                # 1, max_new_tokens
                output = output[:, input_length:]
                if accelerator.num_processes > 1:
                    # pad across device to the same length
                    output = accelerator.pad_across_processes(output.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
                    # num_device, max_new_tokens
                    output = accelerator.gather_for_metrics(output)
                    index = accelerator.gather_for_metrics(index)
                
                output = output.tolist()
                index = index.tolist()

                if accelerator.process_index == 0:
                    pred = tokenizer.batch_decode(output, skip_special_tokens=True)
                    preds.extend(pred)

                    if isinstance(index, list):
                        indices.extend(index)
                    else:
                        # single process
                        indices.append(index)

            if accelerator.process_index == 0:
                scores = []
                for label, pred in tqdm(zip(labels, preds)):
                    # NOTE: here we explicitly input model_name=None
                    score = get_score_one(pred, label, task, None)
                    scores.append(score)
                score = round(sum(scores) / len(scores), 4)

                logger.info(f"{task}: {score}")
                metrics[task] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for index, pred, label in zip(indices, preds, labels):
                        item = {
                            "index": index,
                            "pred": pred,
                            "label": label,
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))

        avg = round(sum(metrics.values()) / len(metrics), 4)
        metrics["avg"] = avg

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
