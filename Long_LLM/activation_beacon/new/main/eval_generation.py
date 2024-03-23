import os
import torch
from typing import List, Optional
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict

from src.data import Data
from src.metrics import Metric
from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, evaluate_generation, split_file_dir_name_ext

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json data.'}
    )
    output_dir: str = field(
        default="data/results/generation/",
        metadata={'help': 'Output directory for results and logs.'}
    )

    min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for evaluation?'}
    )
    max_length: int = field(
        default=100000,
        metadata={'help': 'How many tokens at maximum for evaluation?'}
    )

    seed: int = field(
        default=42
    )
    max_num: int = field(
        default=None,
        metadata={'help': 'Max number of instances to evaluate.'}
    )
    metrics: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'List of metrics. {rouge, save_result}'}
    )


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    result_dir_components = [args.output_dir, "--".join(args.model_name_or_path.strip(os.sep).split(os.sep)[-2:]), str(args.max_length)]
    result_dir = os.path.join(*result_dir_components)

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)

    with accelerator.main_process_first():
        dataset = Data.prepare_eval_data(
            args.eval_data, 
            tokenizer=tokenizer,
            max_length=args.max_length,
            min_length=args.min_length,
            chat_template=args.chat_template,
            seed=args.seed,
            max_eval_num=args.max_num,
            cache_dir=args.dataset_cache_dir,
        )

    # get labels (the target generation result)
    labels = dataset["labels"]
    dataset = dataset.remove_columns(["labels"])

    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        # only pin memory when no gpu
        pin_memory=not args.cpu,
    )

    compute_metrics_fn = Metric.get_metric_fn(
        metrics=args.metrics, 
        save_path=Metric.get_save_path(
            args.eval_data,
            result_dir
        )
    )
    indices, outputs = evaluate_generation(
        model, 
        dataloader, 
        accelerator=accelerator, 
        tokenizer=tokenizer,
    )
    
    if accelerator.process_index == 0:
        metrics = compute_metrics_fn(outputs, labels, indices=indices)

        file_name = split_file_dir_name_ext(args.eval_data)[1]
        log_path = os.path.join(args.output_dir, f"{file_name}.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
