import os
import copy
import json
import logging
import datasets
from typing import List
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from src.lm import (
    LM, 
    LMArgs
)
from src.retrieval import (
    RetrievalArgs, 
    RetrievalMetric,
)
from src.utils.util import makedirs, remove_eos, DefaultDataCollator, DatasetProcessFn, FileLogger
from .eval_retrieval import main as retrieval_main

logger = logging.getLogger(__name__)
import transformers
transformers.logging.set_verbosity_error()

SUBJECT_2_CATEGORY={"abstract_algebra": "STEM", "anatomy": "others", "astronomy": "STEM", "business_ethics": "others", "clinical_knowledge": "others", "college_biology": "STEM", "college_chemistry": "STEM", "college_computer_science": "STEM", "college_mathematics": "STEM", "college_medicine": "others", "college_physics": "STEM", "computer_security": "STEM", "conceptual_physics": "STEM", "econometrics": "Social Sciences", "electrical_engineering": "STEM", "elementary_mathematics": "STEM", "formal_logic": "Humanities", "global_facts": "others", "high_school_biology": "STEM", "high_school_chemistry": "STEM", "high_school_computer_science": "STEM", "high_school_european_history": "Humanities", "high_school_geography": "Social Sciences", "high_school_government_and_politics": "Social Sciences", "high_school_macroeconomics": "Social Sciences", "high_school_mathematics": "STEM", "high_school_microeconomics": "Social Sciences", "high_school_physics": "STEM", "high_school_psychology": "Social Sciences", "high_school_statistics": "STEM", "high_school_us_history": "Humanities", "high_school_world_history": "Humanities", "human_aging": "others", "human_sexuality": "Social Sciences", "international_law": "Humanities", "jurisprudence": "Humanities", "logical_fallacies": "Humanities", "machine_learning": "STEM", "management": "others", "marketing": "others", "medical_genetics": "others", "miscellaneous": "others", "moral_disputes": "Humanities", "moral_scenarios": "Humanities", "nutrition": "others", "philosophy": "Humanities", "prehistory": "Humanities", "professional_accounting": "others", "professional_law": "Humanities", "professional_medicine": "others", "professional_psychology": "Social Sciences", "public_relations": "Social Sciences", "security_studies": "Social Sciences", "sociology": "Social Sciences", "us_foreign_policy": "Social Sciences", "virology": "others", "world_religions": "Humanities"}


@dataclass
class MMLUArgs(LMArgs, RetrievalArgs):
    output_dir: str = field(
        default="data/results/mmlu",
    )
    eval_data: str = field(
        default="llm-embedder:qa/mmlu/test.json",
        metadata={'help': 'Path to the test file.'}
    )
    lm_batch_size: int = field(
        default=2,
        metadata={'help': 'Evaluation batch size.'},
    )

    few_shot: int = field(
        default=0,
        metadata={'help': 'How many few shot train samples?'},
    )
    train_data: str = field(
        default="llm-embedder:qa/mmlu/dev.json",
        metadata={'help': 'Path to the file containing training examples.'}
    )

    corpus: str = field(
        default="llm-embedder:qa/msmarco/corpus.json",
        metadata={'help': 'Corpus path for retrieval.'}
    )
    key_template: str = field(
        default="{title} {text}",
        metadata={'help': 'How to concatenate columns in the corpus to form one key?'}
    )
    key_max_length: int = field(
        default=128,
        metadata={'help': 'How many tokens at maximum in a key.'}
    )
    hits: int = field(
        default=10,
        metadata={'help': 'How many hits per query?'},
    )
    key_num: int = field(
        default=3,
        metadata={'help': 'How many docs to provide in prompt?'},
    )
    metrics: List[str] = field(
        default_factory=lambda: ["collate_key"],
    )
    save_to_output: bool = field(
        default=True,
        metadata={'help': 'Save the result/key/negative to output_dir? If not true, they will be saved next to the eval_data.'}
    )

    log_path: str = field(
        default="data/results/mmlu/mmlu.log",
        metadata={'help': 'Path to the file for logging.'}
    )
    

def process_mmlu(tokenizer, context_max_length=2048, key_num=3, few_shot=0, train_data=None, cache_dir=None, is_encoder_decoder=False, add_llama_inst=False):
    tokenizer.truncation_side = 'right'
    left_truncation_tokenizer = copy.deepcopy(tokenizer)
    left_truncation_tokenizer.truncation_side = 'left'

    test = tokenizer("test", return_special_tokens_mask=True)["special_tokens_mask"]

    has_bos = has_eos = False
    if test[0] == 1:
        has_bos = True
    if test[-1] == 1:
        has_eos = True
    
    if few_shot > 0:
        assert train_data is not None
        train_data = datasets.load_dataset("json", data_files=train_data, cache_dir=cache_dir, split="train")
        train_df = train_data.to_pandas()
        # transform the dataframe into dict of dataframes
        train_df = {k: v[:few_shot] for k, v in train_df.groupby("subject")}
        
    options = ['A', 'B', 'C', 'D']
    
    def _prepare_sample(query, choices, answer):
        """
        <Question>
        A. <Choices 1>
        B. <Choices 2>
        C. <Choices 3>
        D. <Choices 4>
        Answer: <Answer>
        """
        # answer maybe int or numpy int64
        if not isinstance(answer, str):
            answer = options[answer]

        sample = f"{query}\n{chr(10).join([f'{option}. {choice}' for option, choice in zip(options, choices)])}\nAnswer: {answer}"
        return sample
    
    def _prepare_knowledge(key, max_length=None):
        if key is not None:
            key = key[:key_num]
            key = "\n".join(key)
            key = f"Knowledge:\n{key}"
            if max_length is not None:
                # truncate key if necessary
                key = tokenizer.decode(tokenizer.encode(key, add_special_tokens=False, truncation=True, max_length=max_length))
        else:
            key = ""
        return key

    @DatasetProcessFn(augment=True)
    def _process(query, choices, query_id, subject, answer, key=None, **kwds):
        """Yield key and query with a prompt template"""
        output = defaultdict(list)
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

        knowledge_max_length = context_max_length - len(tokenizer.encode(head + train_samples + _prepare_sample(query, choices, 'A'))) - int(has_bos) - int(has_eos)
        if knowledge_max_length < 0:
            knowledge = ""
        else:
            knowledge = _prepare_knowledge(key, knowledge_max_length)

        for option in options:
            left = knowledge
            right = head + train_samples + _prepare_sample(query, choices, option)
            # \n\n to split knowledge and prompts
            if len(left):
                right = "\n\n" + right

            # TODO: add llama instruction
            # if add_llama_inst:
            #     left = "[INST]" + left
            #     right = right + "[/INST]"

            inputs = left_truncation_tokenizer(left + right, truncation=True, max_length=context_max_length, return_token_type_ids=False)

            if has_eos and not is_encoder_decoder:
                inputs = remove_eos(inputs, tokenizer.eos_token_id)

            # find answer length
            option_seq = tokenizer.encode("Answer: " + option, add_special_tokens=False)            
            option_length = len(option_seq) - len(tokenizer.encode("Answer:", add_special_tokens=False))

            if is_encoder_decoder:
                labels = inputs["input_ids"].copy()[-option_length:]
                for k, v in inputs.items():
                    inputs[k] = v[:-option_length]
                inputs["labels"] = labels

            else:
                # take care of padded tokens
                labels = inputs["input_ids"].copy()
                labels = [x if inputs["attention_mask"][i] == 1 else -100 for i, x in enumerate(labels)]
                labels[:-option_length] = [-100] * (len(labels) - option_length)
                inputs["labels"] = labels

            inputs["query_id"] = query_id
            for k, v in inputs.items():
                output[k].append(v)
        return output
    return _process


def evaluate_mmlu(eval_data, save_path, **kwds):
    def compute_metric(eval_preds):
        makedirs(save_path)

        tasks = defaultdict(list)
        results = defaultdict(list)
        samples = {}
        
        with open(eval_data) as f:
            for line in f:
                sample = json.loads(line.strip())
                samples[sample["query_id"]] = sample
        
        # nll must comes in the order of A, B, C, and D
        for query_id, nll in zip(*eval_preds):
            # store log likelihood
            results[query_id].append(-nll)
        
        with open(makedirs(save_path), "w") as f:
            for k, v in results.items():
                output = max(enumerate(v), key=lambda x: x[1])[0]
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
        
        metrics = {
            "STEM": metrics["STEM"],
            "Social Sciences": metrics["Social Sciences"],
            "Humanities": metrics["Humanities"],
            "Others": metrics["others"],
            "All": metrics["all"],
        }

        return dict(metrics)
    return compute_metric


def main():
    parser = HfArgumentParser([MMLUArgs])
    args, = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(cpu=args.cpu)

    # modify the output_dir for retrieval
    if args.retrieval_method == "dense":
        output_dir = os.path.join(args.output_dir, args.query_encoder.strip(os.sep).replace(os.sep, "--"))
    else:
        output_dir = os.path.join(args.output_dir, args.retrieval_method)
    args.output_dir = output_dir

    if args.retrieval_method != "no":
        retrieval_main(args=args, accelerator=accelerator, log=False)
        eval_data = RetrievalMetric._get_save_path(args.eval_data, args.output_dir, field="key", save_name=args.save_name)
    else:
        eval_data = args.eval_data

    lm = LM(
        model_name_or_path=args.model_name_or_path,
        dtype=args.lm_dtype,
        device_map=args.lm_device_map,
        padding_side=args.padding_side,
        cache_dir=args.model_cache_dir,
        accelerator=accelerator
    )
    
    tokenizer = lm.tokenizer

    with accelerator.main_process_first():
        logging.info(f"Loading data from {eval_data}...")
        dataset = datasets.load_dataset("json", data_files=eval_data, split="train", cache_dir=args.dataset_cache_dir)
        dataset = dataset.map(process_mmlu(
            tokenizer, 
            context_max_length=args.context_max_length, 
            key_num=args.key_num,
            few_shot=args.few_shot,
            train_data=args.train_data,
            cache_dir=args.dataset_cache_dir,
            is_encoder_decoder=lm.model.config.is_encoder_decoder,
            add_llama_inst=args.add_llama_inst
        ), remove_columns=dataset.column_names, batched=True, num_proc=32)

    data_collator = DefaultDataCollator(tokenizer=tokenizer, add_position_ids=args.add_position_ids)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.lm_batch_size, 
        collate_fn=data_collator,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    results = lm.compute_nlls(dataloader)

    if accelerator.process_index == 0:
        file_logger = FileLogger(makedirs(args.log_path))    
        result_path = os.path.join(args.output_dir, args.model_name_or_path.strip(os.sep).replace(os.sep, "--") + ".json")
        metrics = evaluate_mmlu(eval_data, result_path)(results)
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
