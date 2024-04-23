import re
import os
import json
import torch
import datasets
import numpy as np
from typing import List
from tqdm import tqdm
from fuzzywuzzy import fuzz
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from dataclasses import dataclass, field, asdict

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/passkey/",
        metadata={'help': 'Output directory for results and logs.'}
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

    passkey_length: int = field(
        default=5,
        metadata={'help': 'How many numbers are in the passkey?'}
    )
    seed: int = field(
        default=123,
        metadata={'help': 'Random seed.'}
    )



def generate_sample(tokenizer, chat_template, context_length, passkey_depth, passkey_length, rng:np.random.Generator=np.random.default_rng(42)):
    passkey = str(rng.integers(10**(passkey_length - 1), 10**passkey_length))
    description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"
    noises = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." * (context_length // 10)
    information = f"\n\nThe pass key is {passkey}. Remember it. {passkey} is the pass key.\n\n"
    prompt = "What is the pass key?"

    # these inputs are used only once
    description_input_ids = tokenizer.encode(description)
    information_input_ids = tokenizer.encode(information, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    information_length = len(information_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - information_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    passkey_pos = minimum_pos + round((maximum_pos - minimum_pos) * passkey_depth / 100)

    # DEBUG
    # information_pos = description_length
    # information_pos = rng.integers(minimum_pos, min(maximum_pos, 1000))
    # information_pos = rng.integers(1024, min(maximum_pos, 2000))

    prefix_noise = tokenizer.encode(noises, max_length=passkey_pos - description_length, truncation=True, add_special_tokens=False)
    suffix_noise = tokenizer.encode(noises, max_length=context_length - passkey_pos - information_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_input_ids, prefix_noise, information_input_ids, suffix_noise, prompt_input_ids], [])
    inputs = tokenizer.decode(input_ids)

    if chat_template != "no":
        inputs = apply_chat_template(chat_template, messages=[{'role': 'user', 'content': inputs}], add_generation_prompt=True)

    return inputs, prompt, passkey


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

    rng_state = np.random.default_rng(args.seed)

    all_inputs = []
    for length in tqdm(test_lengths, desc="Constructing Data"):
        for depth in test_depths:
            inputs, prompt, passkey = generate_sample(
                tokenizer=tokenizer, 
                chat_template=args.chat_template, 
                context_length=length, 
                passkey_depth=depth, 
                passkey_length=args.passkey_length,
                rng=rng_state
            )
            all_inputs.append({'inputs': inputs, 'prompt': prompt, 'passkey': passkey, 'length': length, 'depth': depth})

    dataset = datasets.Dataset.from_list(all_inputs)
    dataloader = torch.utils.data.DataLoader(
        # length and depth are useless in forward computation
        dataset.remove_columns(['length', 'depth', 'passkey']), 
        batch_size=args.batch_size, 
        collate_fn=DefaultDataCollator(tokenizer),
        pin_memory=not args.cpu,
    )
    dataloader = accelerator.prepare(dataloader)

    accelerator.wait_for_everyone()

    accuracy = {l: {d: [] for d in test_depths} for l in test_lengths}
    fuzzy_score = {l: {d: [] for d in test_depths} for l in test_lengths}
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

        for l, d, p, o in zip(dataset['length'], dataset['depth'], dataset['passkey'], all_outputs):
            # extract numbers
            o = re.search("\d+", o)
            if o:
                o = o.group()
            else:
                o = ""

            acc = float(p == o)
            score = round(fuzz.ratio(o, p), 2)

            accuracy[l][d].append(acc)
            fuzzy_score[l][d].append(score)
            results[l][d].append({'target': p, 'prediction': o})

        for l, lv in accuracy.items():
            for d, dv in lv.items():
                accuracy[l][d] = round(sum(dv) / len(dv), 2)
        for l, lv in fuzzy_score.items():
            for d, dv in lv.items():
                fuzzy_score[l][d] = round(sum(dv) / len(dv), 2)
        
        result_dir_components = [args.output_dir, "--".join(args.model_name_or_path.strip(os.sep).split(os.sep)[-2:])]
        if hasattr(model, "memory"):
            result_dir_components.append(f"{model.memory.beacon_ratio}")
        result_dir = os.path.join(*result_dir_components)
        with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
            json.dump(results, f)

        log_path = os.path.join(args.output_dir, f"metrics.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log({'accuracy': accuracy, 'fuzz': fuzzy_score}, Args=asdict(args))


if __name__ == "__main__":
    main()
