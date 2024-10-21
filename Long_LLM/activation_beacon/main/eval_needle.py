import os
import math
import torch
import json
import datasets
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rouge import Rouge
from glob import glob
from typing import List, Optional
from tqdm import tqdm
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from dataclasses import dataclass, field, asdict

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    haystack_path: str = field(
        default="long-llm:needle/PaulGrahamEssays",
        metadata={'help': 'The context for evaluation.'}
    )
    output_dir: str = field(
        default="data/results/needle/",
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
        default=10,
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

    needle: str = field(
        default="\n\nThe best thing to do in San Francisco is sitting in Dolores Park and eating a hamburg on a sunny day.\n\n",
        metadata={'help': 'The needle content'}
    )
    prompt: str = field(
        default='\n\nWhat is the best thing to do in San Francisco?\nAnswer:',
        metadata={'help': 'The needle content'}
    )

    gpt_eval: bool = field(
        default=False,
        metadata={'help': 'Use GPT4 to evaluate accuracy.'}
    )

    load_result: bool = field(
        default=False,
        metadata={'help': 'Load previous results?'}
    )

    do_sample: bool = False
    max_new_tokens: int = 50

    def __post_init__(self):
        super().__post_init__()
        self.haystack_path = self.resolve_path(self.haystack_path)


class OpenAIEvaluator:
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """
        from langchain_openai import ChatOpenAI
        # from langchain_community.chat_models import ChatOpenAI

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv('OPENAI_API_KEY')
        if (not api_key):
            raise ValueError("OPENAI_API_KEY must be in env for using openai evaluator.")
        proxy = os.getenv('http_proxy')
        if proxy:
            logger.info(f"Using proxy {proxy}...")

        self.evaluator = ChatOpenAI(model=self.model_name,
                                    openai_api_key=api_key,
                                    openai_proxy=proxy,
                                    **self.model_kwargs)

    def evaluate_response(self, response: str) -> int:
        from langchain.evaluation import load_evaluator

        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])


def generate_sample(
    tokenizer, 
    chat_template, 
    context, 
    context_length, 
    needle_depth, 
    needle="\n\nThe best thing to do in San Francisco is sitting in Dolores Park and eating a hamburg on a sunny day.\n\n", 
    prompt='\n\nWhat is the best thing to do in San Francisco?\nAnswer:'
):
    num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
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

    inputs = apply_chat_template(chat_template, messages=[{'role': 'user', 'content': inputs}], tokenizer=tokenizer, add_generation_prompt=True).raw

    return inputs, prompt, needle


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)

    result_dir = os.path.join(args.output_dir, args.result_dir)

    if args.load_result:
        with open(makedirs(os.path.join(result_dir, "results.json")), "r", encoding='utf-8') as f:
            results = json.load(f)

    else:
        model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

        if args.test_length is None:
            test_lengths = np.linspace(args.min_length, args.max_length, args.num_length_interval, endpoint=True).astype(int).tolist()
        else:
            test_lengths = args.test_length

        if args.test_depth is None:
            test_depths = np.linspace(args.min_depth, args.max_depth, args.num_depth_interval, endpoint=True).astype(int).tolist()
        else:
            test_depths = args.test_depth

        if os.path.isfile(args.haystack_path):
            with open(args.haystack_path) as f:
                context = f.read().strip()
        elif os.path.isdir(args.haystack_path):
            context = ""
            num_tokens = 0
            for file in glob(f"{args.haystack_path}/*.txt"):
                with open(file, 'r') as f:
                    this_file_context = f.read()
                    num_tokens += len(tokenizer.encode(this_file_context, add_special_tokens=False))
                    context += this_file_context
                    if num_tokens > max(test_lengths):
                        break
        else:
            raise ValueError(f"Cannot find haystack: {args.haystack_path}")

        all_inputs = []
        for length in tqdm(test_lengths, desc="Constructing Data"):
            for depth in test_depths:
                inputs, prompt, needle = generate_sample(
                    tokenizer=tokenizer, 
                    chat_template=args.chat_template, 
                    context=context,
                    context_length=length, 
                    needle_depth=depth,
                    needle=args.needle,
                    prompt=args.prompt
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
            results = {l: {d: [] for d in test_depths} for l in test_lengths}

            all_lengths = dataset['length']
            all_depths = dataset['depth']
            all_needles = dataset['needle']

            for l, d, n, o in zip(all_lengths, all_depths, all_needles, all_outputs):
                results[l][d].append({'target': n, 'prediction': o})

            with open(makedirs(os.path.join(result_dir, "results.json")), "w", encoding='utf-8') as f:
                json.dump(results, f)
            # also save config
            args.save(os.path.join(result_dir, "config.json"))

    if accelerator.process_index == 0:
        rouge = Rouge()
        rouge_score = {l: {d: [] for d in v.keys()} for l, v in results.items()}
        if args.gpt_eval:
            evaluator = OpenAIEvaluator(question_asked=args.prompt.strip(), true_answer=args.needle.strip())
            gpt_score = {l: {d: [] for d in v.keys()} for l, v in results.items()}

        for l, lv in results.items():
            for d, dv in lv.items():
                for v in dv:
                    prediction = v["prediction"].strip("\n").split("\n")[0]
                    target = v["target"].strip("\n")

                    try:
                        score = rouge.get_scores([prediction], [target], avg=True)["rouge-l"]["r"]
                    except:
                        score = 0

                    rouge_score[l][d].append(score)

                    if args.gpt_eval:
                        while 1:
                            try:
                                gpt_score[l][d].append(evaluator.evaluate_response(prediction))
                                break
                            except ValueError:
                                pass

                rouge_score[l][d] = round(sum(rouge_score[l][d]) / len(dv), 2)
                if args.gpt_eval:
                    while 1:
                        try:
                            gpt_score[l][d] = round(sum(gpt_score[l][d]) / len(dv), 2)
                            break
                        except ValueError:
                            pass

        metrics = {'rouge': rouge_score}
        if args.gpt_eval:
            metrics["gpt"] = gpt_score
        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))

        for metric_key, metric_value in metrics.items():
            # Copied from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb
            cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
            # Create the heatmap with better aesthetics
            sns.set(rc={"figure.figsize": (17.5, 8), "axes.titlesize":14, "axes.labelsize":12}, style="whitegrid", palette="colorblind")
            data = pd.DataFrame(metric_value)

            if metric_key == "rouge":
                vmin = 0
                vmax = 1.0
                label = "Rouge"
            elif metric_key == "gpt":
                vmin = 1
                vmax = 10.0
                label = "Accuracy"

            annot = data.copy().astype(str)
            annot[annot == str(vmax)] = ""

            ax = sns.heatmap(
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annot=annot,
                fmt="",
                linewidth=.5,
                annot_kws={"fontsize":10},
            )
            cbar = ax.collections[0].colorbar
            cbar.set_label(label, size=14)

            # More aesthetics
            plt.title('Needle In A HayStack')  # Adds a title
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
