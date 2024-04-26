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

from src import ModelArgs, DatasetProcessFn, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs
from .longbench_utils import DATASET2PROMPT, DATASET2MAXNEWTOKENS, DATASET2CATEGORY, scorer, scorer_e

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="activation-beacon:longbench/test.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/longbench/",
        metadata={'help': 'Output directory for results and logs.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    dataset_names: List[str] = field(
        default_factory=lambda: ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum', 'lcc', 'repobench-p'],
        metadata={'help': 'Which dataset to evaluate?'}
    )

    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={'help': 'Model name on huggingface.'}
    )
    max_length: int = field(
        default=3500,
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



def process_longbench(tokenizer, prompt_templates:Optional[Dict]=None, max_length=3500, add_chat_inst=False, truncate_from_middle=True):
    @DatasetProcessFn()
    def _process(input:str, context:str, dataset:str, all_classes:Optional[List], answers:List[str], length:int, _index:int, **kwds):        
        output = {}

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

        # chat models are better off without build prompts on these tasks
        if add_chat_inst:
            prompt = f"[INST]{prompt}[/INST]"

        output = tokenizer(prompt, padding=False, truncation=False)
        output["dataset"] = dataset
        output["idx"] = _index
        return output
    return _process


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    if ".e." in args.eval_data:
        args.output_dir = args.output_dir.replace("longbench", "longbench_e")
    else:
        args.output_dir = args.output_dir.replace("longbench_e", "longbench")

    result_dir_components = [args.output_dir, args.model_name_or_path.strip(os.sep).replace(os.sep, "--"), str(args.max_length)]
    result_dir = os.path.join(*result_dir_components)

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)

    with accelerator.main_process_first():
        process_fn = process_longbench(
            tokenizer,
            max_length=args.max_length,
            prompt_templates=DATASET2PROMPT,
            add_chat_inst=args.add_chat_inst,
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
            dataloader = accelerator.prepare(dataloader)

            indices = []
            preds = []
            max_new_tokens = DATASET2MAXNEWTOKENS[dataset_name]

            for i, x in enumerate(tqdm(dataloader, desc="Generating")):
                x.pop("dataset")
                idx = x.pop("idx")[0]
                input_length = x["input_ids"].shape[1]

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory") and model.memory is not None:
                    model.memory.reset()

                # NOTE: very important to include \n as an eos token for QA and trec, otherwise the F1 score is devastating
                if dataset_name in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
                    output = model.generate(
                        **x,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                        # prevent warning
                        temperature=1.0,
                        top_p=1.0,
                    )
                else:
                    output = model.generate(
                        **x,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                    )

                # 1, max_new_tokens
                output = output[:, input_length:]
                # pad across device to the same length
                output = accelerator.pad_across_processes(output.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
                # num_device, max_new_tokens
                output = accelerator.gather_for_metrics(output)
                idx = accelerator.gather_for_metrics(idx).tolist()

                if accelerator.process_index == 0:
                    pred = tokenizer.batch_decode(output, skip_special_tokens=True)
                    preds.extend(pred)
                    if isinstance(idx, list):
                        indices.extend(idx)
                    else:
                        # single process
                        indices.append(idx)

            if accelerator.process_index == 0:
                raw_dataset_subset = raw_dataset[indices]
                answers = raw_dataset_subset["answers"]
                lengths = raw_dataset_subset["length"]
                all_classes = raw_dataset_subset["all_classes"][0]
                if '.e.' in args.eval_data:
                    score = scorer_e(dataset_name, preds, answers, lengths, all_classes)
                else:
                    score = scorer(dataset_name, preds, answers, all_classes)        
                
                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for idx, pred in zip(indices, preds):
                        sample = raw_dataset[idx]
                        del sample["all_classes"]
                        del sample["context"]
                        del sample["language"]
                        del sample["_id"]
                        sample["pred"] = pred
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        log_path = os.path.join(args.output_dir, "metrics.log")

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

        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args), Category_Metrics=category_metrics)


if __name__ == "__main__":
    main()
