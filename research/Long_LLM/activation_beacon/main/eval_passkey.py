import re
import os
import json
import torch
import datasets
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


from typing import List, Optional
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
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    min_length: int = field(
        default=8192,
        metadata={'help': 'Minimum context length in evaluation.'}
    )
    max_length: int = field(
        default=131072,
        metadata={'help': 'Maximum context length in evaluation.'}
    )
    num_length_interval: int = field(
        default=20,
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
        default=10,
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

    do_sample: bool = False
    max_new_tokens: int = 50


def generate_sample(tokenizer, chat_template, context_length, passkey_depth, passkey_length, rng:np.random.Generator=np.random.default_rng(42)):
    passkey = str(rng.integers(10**(passkey_length - 1), 10**passkey_length))
    description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"
    noises = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again." * (context_length // 10)
    information = f"\n\nThe pass key is {passkey}. Remember it. {passkey} is the pass key.\n\n"
    prompt = "\n\nWhat is the pass key?"

    # these inputs are used only once
    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
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

    inputs = apply_chat_template(chat_template, messages=[{'role': 'user', 'content': inputs}], tokenizer=tokenizer, add_generation_prompt=True).raw

    return inputs, prompt, passkey


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

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

    # NOTE: prepare dataloader so the data moves to GPU automatically
    dataloader = accelerator.prepare(dataloader)

    accelerator.wait_for_everyone()

    all_outputs = []

    for x in tqdm(dataloader, desc="Evaluating"):
        prompt = x.pop("prompt")
        inputs = x.pop("inputs")
        # TODO: retrieval

        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

        output = model.generate(**inputs)

        if isinstance(output, torch.Tensor):
            # 1, max_new_tokens
            output = output[:, inputs['input_ids'].shape[1]:]
            output = tokenizer.batch_decode(output, skip_special_tokens=True)
        elif isinstance(output, list):
            pass

        if accelerator.num_processes > 1:
            output = accelerator.gather_for_metrics(output)

        all_outputs.extend(output)

    if accelerator.process_index == 0:
        accuracy = {l: {d: [] for d in test_depths} for l in test_lengths}
        fuzzy_score = {l: {d: [] for d in test_depths} for l in test_lengths}
        results = {l: {d: [] for d in test_depths} for l in test_lengths}

        for l, d, p, o in zip(dataset['length'], dataset['depth'], dataset['passkey'], all_outputs):
            # extract numbers
            o = re.search("\d+", o)
            if o:
                o = o.group()
            else:
                o = ""
            results[l][d].append({'target': p, 'prediction': o})

            acc = float(p == o)
            score = round(fuzz.ratio(o, p) / 100, 2)

            accuracy[l][d].append(acc)
            fuzzy_score[l][d].append(score)

        for l, lv in accuracy.items():
            for d, dv in lv.items():
                accuracy[l][d] = round(sum(dv) / len(dv), 2)

        for l, lv in fuzzy_score.items():
            for d, dv in lv.items():
                fuzzy_score[l][d] = round(sum(dv) / len(dv), 2)
        
        result_dir = os.path.join(args.output_dir, args.result_dir)
        with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
            json.dump(results, f)
        # also save config
        args.save(os.path.join(result_dir, "config.json"))

        metrics = {'accuracy': accuracy, 'fuzz': fuzzy_score}
        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))

        for metric_key, metric_value in metrics.items():
            # Copied from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
            # Create the heatmap with better aesthetics
            sns.set(rc={"figure.figsize": (17.5, 8), "axes.titlesize":14, "axes.labelsize":12}, style="whitegrid", palette="colorblind")
            data = pd.DataFrame(metric_value)
            ax = sns.heatmap(
                data,
                cmap=cmap,
                vmin=0,
                vmax=1,
                fmt="g",
                linewidth=.5,
            )
            cbar = ax.collections[0].colorbar
            cbar.set_label(metric_key, size=14)

            # More aesthetics
            plt.title('Passkey Retrieval')  # Adds a title
            plt.xlabel('Context Length', fontsize=14)  # X-axis label
            plt.ylabel('Depth Percent', fontsize=14)  # Y-axis label
            plt.xticks(rotation=45, fontsize=10)  # Rotates the x-axis labels to prevent overlap
            plt.yticks(rotation=0, fontsize=10)  # Ensures the y-axis labels are horizontal
            plt.tight_layout()  # Fits everything neatly into the figure area
            # save to result_dir
            plt.savefig(os.path.join(result_dir, f"{metric_key}.png"), format='png', bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    main()
