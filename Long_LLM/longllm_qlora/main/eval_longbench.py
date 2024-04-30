import os
import datasets
import json
import torch
from tqdm import tqdm
from typing import Optional, Dict, List
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
        default="long-llm:longbench/test.json",
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

    dataset_names: List[str] = field(
        default_factory=lambda: ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum', 'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p'],
        metadata={'help': 'Which dataset to evaluate?'}
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


def process_longbench(tokenizer, chat_template, prompt_templates:Optional[Dict]=None, max_length=3500, truncate_from_middle=True):
    def _process(data, indices):
        outputs = {'input_ids': [], 'attention_mask': [], "dataset": [], "index": []}

        for input, context, dataset, index in zip(data['input'], data['context'], data['dataset'], indices):
            if dataset.endswith("_e"):
                dataset = dataset[:-2]
            
            prompt_template = prompt_templates[dataset]
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
            if not any(x in DATASET2CATEGORY[dataset] for x in ["Few-Shot Learning", "Code Completion"]):
                prompt = apply_chat_template(
                    chat_template, 
                    messages=[{'role': 'user', 'content': prompt}],
                    tokenizer=tokenizer,
                    add_generation_prompt=True,
                ).raw

            encoded = tokenizer(prompt)

            for k, v in encoded.items():
                outputs[k].append(v)
            outputs["dataset"].append(dataset)
            outputs["index"].append(index)

        return outputs
    return _process


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    # stop generation for QA tasks when \n appears
    if hasattr(model, "generation_config"):
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

    with accelerator.main_process_first():
        process_fn = process_longbench(
            tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            prompt_templates=DATASET2PROMPT,
            truncate_from_middle=args.truncate_from_middle,
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, with_indices=True, remove_columns=raw_dataset.column_names)

    groupby_dataset = dataset.to_pandas().groupby("dataset")

    metrics = {}
    if args.dataset_names is None:
        dataset_names = [key for key, _ in groupby_dataset]
    else:
        dataset_names = args.dataset_names

    result_dir = os.path.join(args.output_dir, args.result_dir)
    for i, dataset_name in enumerate(dataset_names):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")

        result_path = os.path.join(result_dir, f"{dataset_name}.json")
        
        if args.load_result and os.path.exists(result_path):
            if accelerator.process_index == 0:
                with open(result_path, encoding="utf-8") as f:
                    score = json.loads(f.readline())
                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

        else:
            dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)

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
            max_new_tokens = DATASET2MAXNEWTOKENS[dataset_name]

            for i, x in enumerate(tqdm(dataloader, desc="Generating")):
                x.pop("dataset")
                index = x.pop("index")[0]
                input_length = x["input_ids"].shape[1]

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory"):
                    model.memory.reset()

                # NOTE: very important to include \n as an eos token for QA and trec, otherwise the F1 score is devastating
                if dataset_name in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
                    output = model.generate(
                        **x,
                        max_new_tokens=max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=eos_token_id,
                        begin_suppress_tokens=eos_token_id,
                        # FIXME: sometimes transformers cannot detect deepspeed zero3, dont know why
                        synced_gpus=accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3,
                    )
                else:
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
                raw_dataset_subset = raw_dataset[indices]
                answers = raw_dataset_subset["answers"]
                lengths = raw_dataset_subset["length"]
                all_classes = raw_dataset_subset["all_classes"][0]
                score = scorer(dataset_name, preds, answers, all_classes)        
                
                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

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
