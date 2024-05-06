# modified based on https://github.com/dvlab-research/LongLoRA/blob/main/passkey_retrivial.py

import re
import os
import torch
import numpy as np
from typing import List
from tqdm import tqdm
from fuzzywuzzy import fuzz
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from dataclasses import dataclass, field, asdict

from src import ModelArgs, FileLogger, get_model_and_tokenizer, makedirs

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    output_dir: str = field(
        default="data/results/passkey/",
        metadata={'help': 'Output directory for results and logs.'}
    )

    max_length: int = field(
        default=32000,
        metadata={'help': 'How many tokens at maximum?'}
    )
    interval: int = field(
        default=4000,
        metadata={'help': 'How many tokens to increase at a time?'}
    )
    target_length: List[int] = field(
        default=None,
        metadata={'help': 'Specified position to evaluate?'}
    )
    skip: int = field(
        default=0,
        metadata={'help': 'How many test points to skip in the beginning?'}
    )
    passkey_length: int = field(
        default=5,
        metadata={'help': 'How many numbers are in the passkey?'}
    )
    num_tests: int = field(
        default=5,
        metadata={'help': 'How many repetitive test at each interval?'}
    )
    seed: int = field(
        default=123,
        metadata={'help': 'Random seed.'}
    )



def edit_distance(prediction, ground_truth):
    return (fuzz.ratio(prediction, ground_truth) / 100)


def get_inputs(tokenizer, max_length, passkey_length, rng:np.random.Generator):
    """Generates a text file and inserts an passkey at a random position."""
    passkey = str(rng.integers(10**(passkey_length - 1), 10**passkey_length))
    description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n"
    noises = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." * (max_length // 10)
    information = f"\nThe pass key is {passkey}. Remember it. {passkey} is the pass key.\n"
    prompt = "\nWhat is the pass key? The pass key is"

    # these inputs are used only once
    description_inputs = tokenizer.encode(description)
    information_inputs = tokenizer.encode(information, add_special_tokens=False)
    prompt_inputs = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_inputs)
    information_length = len(information_inputs)
    prompt_length = len(prompt_inputs)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = max_length - prompt_length - information_length - 1
    if minimum_pos > max_length or maximum_pos < 0:
        raise ValueError(f"The interval's length {max_length} is too small. Please increase interval!")

    information_pos = rng.integers(minimum_pos, maximum_pos)
    # DEBUG
    # information_pos = rng.integers(0, min(maximum_pos, 1024))
    # information_pos = rng.integers(1024, min(maximum_pos, 2000))

    prefix_noise = tokenizer.encode(noises, max_length=information_pos - description_length, truncation=True, add_special_tokens=False)
    suffix_noise = tokenizer.encode(noises, max_length=max_length - information_pos - information_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_inputs, prefix_noise, information_inputs, suffix_noise, prompt_inputs], [])
    return input_ids, passkey, information_pos


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    assert accelerator.num_processes == 1, "Make sure there is only one process!"
    model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)

    if args.target_length is None:
        total_test_points = args.max_length // args.interval
    else:
        total_test_points = len(args.target_length)
    metrics = {}

    rng_state = np.random.default_rng(args.seed)

    for i in tqdm(range(total_test_points), desc="Evaluating"):
        if i < args.skip:
            continue
        
        passed_tests = 0
        total_tokens = 0
        fuzzy_scores = 0

        if args.target_length is None:
            max_length = (i + 1) * args.interval
        else:
            max_length = args.target_length[i]

        for j in range(args.num_tests):
            input_ids, passkey, information_pos = get_inputs(tokenizer, max_length, args.passkey_length, rng_state)
            # convert to tensor
            input_ids = torch.tensor([input_ids], device=model.device)

            # NOTE: important to reset memory for every batch
            if hasattr(model, "memory") and model.memory is not None:
                model.memory.reset()

            outputs = model.generate(
                input_ids,
                max_new_tokens=args.passkey_length * 2,
                num_beams=1,
                do_sample=False,
                temperature=1.,
            )
            outputs = outputs[0, input_ids.shape[1]:].tolist()

            # extract numbers
            answer = re.search("\d+", tokenizer.decode(outputs, skip_special_tokens=True))
            if answer:
                answer = answer.group()
            else:
                answer = ""

            is_correct = answer == passkey
            fuzzy_scores += fuzz.ratio(answer, passkey)

            passed_tests += is_correct
            total_tokens += input_ids.shape[1]

        avg_tokens = total_tokens // args.num_tests
        accuracy = passed_tests / args.num_tests
        fuzzy_scores = fuzzy_scores / args.num_tests
        metrics[f"Acc-{avg_tokens}"] = accuracy
        metrics[f"Fuzz-{avg_tokens}"] = fuzzy_scores

    if accelerator.process_index == 0:
        log_path = os.path.join(args.output_dir, f"metrics.log")
        file_logger = FileLogger(makedirs(log_path))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
