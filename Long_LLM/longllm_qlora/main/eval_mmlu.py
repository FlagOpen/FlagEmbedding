import os
import copy
import json
import datasets
from typing import List, Optional, Union, Mapping
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from transformers.utils import logging
from dataclasses import dataclass, field
from collections import defaultdict

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template, evaluate_nll, remove_eos

logger = logging.get_logger(__name__)


SUBJECT_2_CATEGORY={"abstract_algebra": "STEM", "anatomy": "others", "astronomy": "STEM", "business_ethics": "others", "clinical_knowledge": "others", "college_biology": "STEM", "college_chemistry": "STEM", "college_computer_science": "STEM", "college_mathematics": "STEM", "college_medicine": "others", "college_physics": "STEM", "computer_security": "STEM", "conceptual_physics": "STEM", "econometrics": "Social Sciences", "electrical_engineering": "STEM", "elementary_mathematics": "STEM", "formal_logic": "Humanities", "global_facts": "others", "high_school_biology": "STEM", "high_school_chemistry": "STEM", "high_school_computer_science": "STEM", "high_school_european_history": "Humanities", "high_school_geography": "Social Sciences", "high_school_government_and_politics": "Social Sciences", "high_school_macroeconomics": "Social Sciences", "high_school_mathematics": "STEM", "high_school_microeconomics": "Social Sciences", "high_school_physics": "STEM", "high_school_psychology": "Social Sciences", "high_school_statistics": "STEM", "high_school_us_history": "Humanities", "high_school_world_history": "Humanities", "human_aging": "others", "human_sexuality": "Social Sciences", "international_law": "Humanities", "jurisprudence": "Humanities", "logical_fallacies": "Humanities", "machine_learning": "STEM", "management": "others", "marketing": "others", "medical_genetics": "others", "miscellaneous": "others", "moral_disputes": "Humanities", "moral_scenarios": "Humanities", "nutrition": "others", "philosophy": "Humanities", "prehistory": "Humanities", "professional_accounting": "others", "professional_law": "Humanities", "professional_medicine": "others", "professional_psychology": "Social Sciences", "public_relations": "Social Sciences", "security_studies": "Social Sciences", "sociology": "Social Sciences", "us_foreign_policy": "Social Sciences", "virology": "others", "world_religions": "Humanities"}


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:mmlu/test.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/mmlu/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    batch_size: int = field(
        default=8,
        metadata={'help': 'Batch size.'}
    )

    few_shot: int = field(
        default=0,
        metadata={'help': 'How many few shot train samples?'},
    )
    train_data: str = field(
        default="long-llm:mmlu/dev.json",
        metadata={'help': 'Path to the file containing training examples.'}
    )


def remove_eos(inputs: Mapping, eos_token_ids: Union[List,int]):
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    input_ids = inputs["input_ids"]
    eos_idx = [i for i, x in enumerate(input_ids) if x in eos_token_ids]
    if len(eos_idx):
        eos_idx = eos_idx[-1]
    else:
        return inputs
    for k, v in inputs.items():
        inputs[k].pop(eos_idx)
    return inputs

def process_mmlu(tokenizer, chat_template, eos_token_id, few_shot=0, train_data=None, cache_dir=None):
    if few_shot > 0:
        assert train_data is not None
        train_data = datasets.load_dataset("json", data_files=train_data, cache_dir=cache_dir, split="train")
        train_df = train_data.to_pandas()
        # transform the dataframe into dict of dataframes
        train_df = {k: v[:few_shot] for k, v in train_df.groupby("subject")}
        
    options = ['A', 'B', 'C', 'D']
    
    def _prepare_sample(query, choices, answer:str=None):
        """
        <Question>
        A. <Choices 1>
        B. <Choices 2>
        C. <Choices 3>
        D. <Choices 4>
        Answer: <Answer>
        """
        # answer maybe int or numpy int64
        if answer is not None and not isinstance(answer, str):
            answer = options[answer]

        option_components = []
        for option, choice in zip(options, choices):
            option_components.append(f'{option}. {choice}')
        option_string = "\n".join(option_components)

        if answer is None:
            sample = f"{query}\n{option_string}\nAnswer:"
        else:
            sample = f"{query}\n{option_string}\nAnswer: {answer}"
        return sample

    def _process(data, indices):
        """Yield key and query with a prompt template"""
        outputs = {"input_ids": [], "attention_mask": [], "labels": [], "index": []}
        
        for index, query, subject, choices, answer in zip(indices, data["query"], data["subject"], data["choices"], data["answer"]):
            query = query.strip()

            head = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

            if few_shot > 0:
                train_samples = ""
                for i in range(few_shot):
                    if i >= len(train_df[subject]):
                        break
                    train_sample = train_df[subject].iloc[i][['query', 'choices', 'answer']]
                    train_sample = _prepare_sample(**train_sample) + "\n\n"
                    train_samples += train_sample
            else:
                train_samples = ""

            for option in options:
                prompt = head + train_samples + _prepare_sample(query, choices)
                answer = option

                encoded = apply_chat_template(
                    chat_template,
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}], 
                    tokenizer=tokenizer, 
                    return_labels=True
                ).encoded

                encoded = remove_eos(encoded, eos_token_id)

                encoded["index"] = index
                for k, v in encoded.items():
                    outputs[k].append(v)

        return outputs
    return _process

def evaluate_mmlu(eval_data, save_path, eval_preds):
    makedirs(save_path)

    tasks = defaultdict(list)
    samples = {}
    
    with open(eval_data) as f:
        for line in f:
            sample = json.loads(line.strip())
            samples[sample["query_id"]] = sample
    
    with open(makedirs(save_path), "w") as f:
        for k, v in eval_preds.items():
            output = min(enumerate(v), key=lambda x: x[1])[0]
            sample = samples[k]
            sample["output"] = output
            tasks[sample["subject"]].append((output, sample["answer"]))
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metrics = defaultdict(list)
    for task_name, task_eval_preds in tasks.items():
        accuracy = 0
        for pred, label in task_eval_preds:
            accuracy += int(pred == label)
        accuracy /= len(task_eval_preds)

        category = SUBJECT_2_CATEGORY[task_name]
        metrics[f"{category}"].append(accuracy)
        metrics["all"].append(accuracy)

    for k, v in metrics.items():
        metrics[k] = sum(v) / len(v)
    
    # for printing
    metrics = {
        "STEM": metrics["STEM"],
        "Social Sciences": metrics["Social Sciences"],
        "Humanities": metrics["Humanities"],
        "Others": metrics["others"],
        "All": metrics["all"],
    }
    return dict(metrics)


def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    result_dir = os.path.join(args.output_dir, args.result_dir)

    eval_data = args.eval_data

    with accelerator.main_process_first():
        dataset = datasets.load_dataset("json", data_files=eval_data, split="train", cache_dir=args.dataset_cache_dir)
        dataset = dataset.map(process_mmlu(
            tokenizer, 
            chat_template=args.chat_template,
            # strip eos
            eos_token_id=model.generation_config.eos_token_id,
            few_shot=args.few_shot,
            train_data=args.train_data,
            cache_dir=args.dataset_cache_dir,
        ), remove_columns=dataset.column_names, batched=True, num_proc=32, with_indices=True)
    
    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    # a dict, key is index, value is negative log likelihood of the answer
    outputs = evaluate_nll(model, dataloader, accelerator)

    if accelerator.process_index == 0:
        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))

        metrics = evaluate_mmlu(eval_data, os.path.join(result_dir, "results.json"), outputs)
        # save config
        args.save(os.path.join(result_dir, "config.json"))
        file_logger.log(metrics, Args=args.to_dict())


if __name__ == "__main__":
    main()
