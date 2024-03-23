import os
import math
import torch
import json
import datasets
import numpy as np
from typing import List
from tqdm import tqdm
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from dataclasses import dataclass, field, asdict

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template
from .longbench_utils import qa_f1_score

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/needle/",
        metadata={'help': 'Output directory for results and logs.'}
    )
    corpus_path: str = field(
        default="data/toy/book-1M.txt",
        metadata={'help': 'The context for evaluation.'}
    )

    min_length: int = field(
        default=4000,
        metadata={'help': 'Minimum context length in evaluation.'}
    )
    max_length: int = field(
        default=64000,
        metadata={'help': 'Maximum context length in evaluation.'}
    )
    num_length_interval: int = field(
        default=6,
        metadata={'help': 'Number of invervals between min_length and max_length.'}
    )
    test_length: List[int] = field(
        default=None,
        metadata={'help': 'Specified evaluation lengths.'}
    )

    min_depth: float = field(
        default=0,
        metadata={'help': 'Minimum pass key depth in the context.'}
    )
    max_depth: float = field(
        default=100,
        metadata={'help': 'Maximum pass key depth in the context.'}
    )
    num_depth_interval: int = field(
        default=6,
        metadata={'help': 'Number of invervals between min_depth and max_depth.'}
    )
    test_depth: List[int] = field(
        default=None,
        metadata={'help': 'Specified evaluation depths.'}
    )



def generate_sample(tokenizer, chat_template, context, context_length, needle_depth):
    num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"
    needle = f"\n\nThe best thing to do in San Francisco is sitting in Dolores Park and eating a hamburg on a sunny day.\n\n"
    prompt = "What is the best thing to do in San Francisco? Don't give information outside the document or repeat your findings."

    description_input_ids = tokenizer.encode(description)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_input_ids, context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:], prompt_input_ids], [])
    inputs = tokenizer.decode(input_ids)

    if chat_template != "no":
        inputs = apply_chat_template(chat_template, messages=[{'role': 'user', 'content': inputs}], add_generation_prompt=True)

    return inputs, prompt, needle


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)

    if args.test_length is None:
        test_lengths = np.linspace(args.min_length, args.max_length, args.num_length_interval, endpoint=True).astype(int).tolist()
    else:
        test_lengths = args.test_length

    if args.test_depth is None:
        test_depths = np.linspace(args.min_depth, args.max_depth, args.num_depth_interval, endpoint=True).astype(int).tolist()
    else:
        test_depths = args.test_depth

    with open(args.corpus_path) as f:
        context = f.read().strip()

    all_inputs = []
    for length in tqdm(test_lengths, desc="Constructing Data"):
        for depth in test_depths:
            inputs, prompt, needle = generate_sample(
                tokenizer=tokenizer, 
                chat_template=args.chat_template, 
                context=context,
                context_length=length, 
                needle_depth=depth
            )
            all_inputs.append({'inputs': inputs, 'prompt': prompt, 'needle': needle, 'length': length, 'depth': depth})

    dataset = datasets.Dataset.from_list(all_inputs)
    dataloader = torch.utils.data.DataLoader(
        # length and depth are useless in forward computation
        dataset.remove_columns(['length', 'depth', 'needle']), 
        batch_size=args.batch_size, 
        collate_fn=DefaultDataCollator(tokenizer),
        pin_memory=not args.cpu,
    )
    dataloader = accelerator.prepare(dataloader)

    accelerator.wait_for_everyone()

    accuracy = {l: {d: [] for d in test_depths} for l in test_lengths}
    f1_score = {l: {d: [] for d in test_depths} for l in test_lengths}
    results = {l: {d: [] for d in test_depths} for l in test_lengths}

    all_outputs = []

    for x in tqdm(dataloader, desc="Evaluating"):
        prompt = x.pop("prompt")
        inputs = x.pop("inputs")
        # TODO: retrieval

        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory") and model.memory is not None:
            model.memory.reset()

        inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=1,
            do_sample=False,
            temperature=1.,
        )
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

        outputs = accelerator.pad_across_processes(outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
        outputs = accelerator.gather_for_metrics(outputs).tolist()
        all_outputs.extend(outputs)


    if accelerator.process_index == 0:
        all_outputs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True)
        
        for l, d, n, o in zip(dataset['length'], dataset['depth'], dataset['needle'], all_outputs):
            acc = float(n == o)
            score = round(qa_f1_score(o, n), 2)
            accuracy[l][d].append(acc)
            f1_score[l][d].append(score)
            results[l][d].append({'target': n, 'prediction': o})

        for l, lv in accuracy.items():
            for d, dv in lv.items():
                accuracy[l][d] = round(sum(dv) / len(dv), 2)
        for l, lv in f1_score.items():
            for d, dv in lv.items():
                f1_score[l][d] = round(sum(dv) / len(dv), 2)

        result_dir_components = [args.output_dir, "--".join(args.model_name_or_path.strip(os.sep).split(os.sep)[-2:])]
        if hasattr(model, "memory"):
            result_dir_components.append(f"{model.memory.beacon_ratio}")
        result_dir = os.path.join(*result_dir_components)
        with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
            json.dump(results, f)

        log_path = os.path.join(args.output_dir, f"metrics.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log({'accuracy': accuracy, 'fuzz': f1_score}, Args=asdict(args))


if __name__ == "__main__":
    main()
