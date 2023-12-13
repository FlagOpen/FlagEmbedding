import os
import re
import json
import random
import logging
import datasets
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from typing import List, Optional
from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import partial
from transformers import DataCollatorWithPadding

from src.lm import LM, LMArgs, GenerationArgs
from src.retrieval import RetrievalArgs
from src.utils.util import makedirs, load_json, FileLogger
from .eval_retrieval import main as retrieval_main
from .icl_utils import flat_options, perplexity_to_choice, compute_scores, _llm_generation_func, _llm_perplexity_func

logger = logging.getLogger(__name__)


CQA = {
    "arc_c":{'method':'perplexity', 'metric':'acc'},
    "arc_e":{'method':'perplexity', 'metric':'acc'},
    "natural_questions":{'method':'generation', 'metric':'em'},
    "cate_name":'CQA'
}
Commonsense = {
    "copa":{'method':'perplexity', 'metric':'acc'},
    "hellaswag":{'method':'perplexity', 'metric':'acc'},
    "piqa":{'method':'perplexity', 'metric':'acc'},
    'cate_name': 'Commonsense'
}
Coreference = {
    "winogrande":{'method':'perplexity', 'metric':'acc'},
    "wsc":{'method':'perplexity', 'metric':'acc'},
    "wsc273":{'method':'perplexity', 'metric':'acc'},
    'cate_name': 'Coreference'
}
Paraphrase = {
    "mrpc":{'method':'perplexity', 'metric':'acc'},
    "paws":{'method':'perplexity', 'metric':'acc'},
    "qqp":{'method':'perplexity', 'metric':'acc'},
    'cate_name': 'Paraphrase'
}
NLI = {
    "rte":{'method':'perplexity', 'metric':'acc'},
    "snli":{'method':'perplexity', 'metric':'acc'},
    "mnli_m":{'method':'perplexity', 'metric':'acc'},
    "mnli_mm":{'method':'perplexity', 'metric':'acc'},
    "qnli":{'method':'perplexity', 'metric':'acc'},
    'cate_name': 'NLI'
}
ReadingComp = {
    "multirc":{'method':'perplexity', 'metric':'f1'},
    "openbookqa":{'method':'perplexity', 'metric':'acc'},
    "boolq":{'method':'perplexity', 'metric':'acc'},
    "squad_v1":{'method':'generation', 'metric':'em'},
    'cate_name': 'ReadingComp'
}
Sentiment = {
    "sentiment140":{'method':'perplexity', 'metric':'acc'},
    "sst2":{'method':'perplexity', 'metric':'acc'},
    "yelp":{'method':'perplexity', 'metric':'acc'},
    'cate_name': 'Sentiment'
}
Data2Text = {
    "common_gen":{'method':'generation', 'metric':'rl'},
    "e2e_nlg":{'method':'generation', 'metric':'rl'},
    "dart":{'method':'generation', 'metric':'rl'},
    'cate_name': 'Data2Text'
}
Summarize = {
    "aeslc":{'method':'generation', 'metric':'rl'},
    "ag_news":{'method':'perplexity', 'metric':'acc'},
    "gigaword":{'method':'generation', 'metric':'rl'},
    'cate_name': 'Summarize'
}
TASK_LIST = [CQA, Commonsense, Coreference, Paraphrase, NLI, ReadingComp, Sentiment, Data2Text, Summarize]
task2cat = {}
for category in TASK_LIST:
    cat_name = category["cate_name"]
    for key, value in category.items():
        if key == "cate_name":
            continue
        task2cat[key] = cat_name


@dataclass
class ICLArgs(LMArgs, RetrievalArgs):
    output_dir: str = field(
        default="data/results/icl/",
        metadata={'help': 'Path to the file for saving embeddings and results.'}
    )
    eval_data: str = field(
        default="llm-embedder:icl/icl/test.json",
        metadata={'help': 'Path to the file containing both retrieved keys and answers.'}
    )
    task_names: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'List of tasks to evaluate.'}        
    )
    load_prev_result: bool = field(
        default=False,
        metadata={'help': 'Load existing results in output_dir?'}
    )

    context_max_length: int = field(
        default=1024,
        metadata={'help': 'Evaluation json file.'},
    )
    few_shot: int = field(
        default=8,
        metadata={'help': 'How many few shot train samples?'},
    )

    corpus: str = field(
        default="llm-embedder:icl/icl/corpus.json",
        metadata={'help': 'Corpus path for retrieval.'}
    )
    key_template: str = field(
        default="{contents}",
        metadata={'help': 'How to concatenate columns in the corpus to form one key?'}
    )
    metrics: List[str] = field(
        default_factory=lambda: [],
    )

    log_path: str = field(
        default="data/results/icl/icl.log",
        metadata={'help': 'Path to the file for logging.'}
    )


@dataclass
class GenerationArgs(GenerationArgs):
    max_new_tokens: int = field(
        default=64,
        metadata={'help': 'Maximum new tokens to generate.'}
    )


def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)


def load_test_data(knn_inxs,
                   test_data, 
                   corpus_data, 
                   filter_diff_task: bool=False,
                   example_num=8,
                   same_task_random=False,
    ):
    dataset = datasets.load_dataset('json', data_files=test_data)['train']
    passage_dataset = datasets.load_dataset('json', data_files=corpus_data)['train']
    
    task_data = defaultdict(list)
    for i, e in enumerate(tqdm(dataset, desc="Organizing Data")):
        query = remove_double_space(e['query'])
        answers = [remove_double_space(x) for x in e['answers']]
        if knn_inxs is not None:
            if filter_diff_task:
                few_shot = []
                rest_passage = []
                for x in knn_inxs[i]:
                    icl_e = passage_dataset[int(x)]
                    # print(icl_e['task_name'], e['task_name'])
                    if icl_e['task_name'][:4] == e['task_name'][:4]:
                        few_shot.append(remove_double_space(icl_e['contents']))
                        if len(few_shot) > example_num: break
                    else:
                        if len(rest_passage) < example_num:
                            rest_passage.append(remove_double_space(icl_e['contents']))
                
                if len(few_shot) < example_num:
                    few_shot.extend(rest_passage)
                    few_shot = few_shot[:example_num]

            else:
                # if task2cat[e['task_name']] == 'Coreference':
                #     candidates = random.sample(knn_inxs[i][:20], example_num)
                # else:
                #     candidates = knn_inxs[i][:example_num]
                candidates = knn_inxs[i][:example_num]
                few_shot = [remove_double_space(passage_dataset[int(x)]['contents']) for x in candidates]
        else:
            few_shot = []
        data = {"query":query, "answers":answers, "few_shot":few_shot}
        if 'options' in e:
            data['options'] = e['options']
        task_data[e['task_name']].append(data)

    if same_task_random:
        task_name_2_idx = defaultdict(list)
        for i, example in enumerate(tqdm(passage_dataset, "Collecting Task Indices")):
            task_name_2_idx[example["task_name"]].append(i)

        for task_name, task_examples in tqdm(task_data.items(), desc="Collecting Same-Task-Random Examples"):
            if task_name in ["mnli_m", "mnli_mm"]:
                corpus_task_name = "mnli"
            else:
                corpus_task_name = task_name

            for i, _ in enumerate(task_examples):
                task_indices = task_name_2_idx[corpus_task_name]
                example_num = min(example_num, len(task_indices))
                # get examples of the same task
                few_shot = [remove_double_space(content) for content in passage_dataset[random.sample(task_indices, example_num)]["contents"]]
                task_data[task_name][i]["few_shot"] = few_shot

    return task_data


def main():
    parser = HfArgumentParser([ICLArgs, GenerationArgs])
    args, generation_args = parser.parse_args_into_dataclasses()
    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])

    if args.retrieval_method == "dense":
        output_dir = os.path.join(args.output_dir, args.query_encoder.strip(os.sep).replace(os.sep, "--"))
    else:
        output_dir = os.path.join(args.output_dir, args.retrieval_method)
    args.output_dir = output_dir

    if args.retrieval_method != "no":
        _, preds, _ = retrieval_main(args=args, accelerator=accelerator, log=False)
    else:
        preds = None

    llm = LM(
        model_name_or_path=args.model_name_or_path,
        dtype=args.lm_dtype,
        device_map=args.lm_device_map,
        padding_side=args.padding_side,
        cache_dir=args.model_cache_dir,
        accelerator=accelerator,
        generation_args=asdict(generation_args)
    )

    tokenizer = llm.tokenizer

    args.output_dir = os.path.join(args.output_dir, args.model_name_or_path.strip(os.sep).replace(os.sep, "--"))

    task_data = load_test_data(preds, test_data=args.eval_data, corpus_data=args.corpus, example_num=args.few_shot, same_task_random=args.retrieval_method == "same-task-random")

    all_results = []
    metrics = {}
    for task_cate in [CQA, Commonsense, Coreference, Paraphrase, NLI, ReadingComp, Sentiment, Data2Text, Summarize]:
        task_results = []
        for task_name, setting in task_cate.items():
            if task_name == 'cate_name': 
                continue
            # skip tasks that are not specified
            if args.task_names is not None and task_name not in args.task_names:
                continue

            save_path = os.path.join(args.output_dir, f'{task_name}.json')

            if args.load_prev_result and os.path.exists(save_path):
                # the first line is the metric
                result = load_json(save_path, lines=True)[0]
                task_results.append(result['metric_value'][setting['metric']])
                all_results.append(result['metric_value'][setting['metric']])
                if accelerator.process_index == 0:
                    logger.info(f"loading existing results from {save_path}...")
                    print(result)
                continue

            test_data = task_data[task_name]
            if accelerator.process_index == 0:
                print(f"------{task_name} ({len(all_results) + 1} / {30})------")

            if setting['metric'] == 'acc':
                assert setting['method'] == 'perplexity'
            if setting['method'] == 'perplexity':
                flat_data = flat_options(test_data)
                dataset = datasets.Dataset.from_list(flat_data)
                dataset.set_transform(
                    partial(
                        _llm_perplexity_func, 
                        tokenizer=tokenizer,
                        example_num=args.few_shot,
                        max_input_tokens=args.context_max_length,
                        add_llama_inst=args.add_llama_inst,
                    )
                )
            else:
                dataset = datasets.Dataset.from_list(test_data)
                dataset.set_transform(
                    partial(
                        _llm_generation_func, 
                        tokenizer=tokenizer,
                        example_num=args.few_shot,
                        max_input_tokens=args.context_max_length,
                        add_llama_inst=args.add_llama_inst,
                    )
                )
            
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=args.lm_batch_size, 
                collate_fn=data_collator,
                pin_memory=True,
            )
            dataloader = accelerator.prepare(dataloader)

            if setting['method'] == 'perplexity':
                predictions = llm.compute_nlls(dataloader)
                predictions = perplexity_to_choice(test_data, predictions)
            else:
                if args.add_llama_inst:
                    eos_token_id = tokenizer.eos_token_id
                else:
                    eos_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

                predictions = llm.generate(dataloader, eos_token_id=eos_token_id)
                predictions = [x.strip() for x in predictions]

            if setting['metric'] in ['em']:
                labels = [x['answers'] for x in test_data]
            else:
                labels = [x['answers'][0] for x in test_data]
            
            metric_value = compute_scores(setting['metric'], predictions, labels)
    
            result = {'task_name':task_name, 'setting':setting, 'metric_value':metric_value}
            if accelerator.process_index == 0:
                print(result)
                with open(makedirs(save_path), 'w') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    for i, sample in enumerate(test_data):
                        sample["output"] = predictions[i]
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            task_results.append(result['metric_value'][setting['metric']])
            all_results.append(result['metric_value'][setting['metric']])

        if len(task_results):
            metrics[task_cate['cate_name']] = np.mean(task_results)

    metrics['avg'] = np.mean(all_results)

    file_logger = FileLogger(makedirs(args.log_path))
    if accelerator.process_index == 0:
        file_logger.log(metrics, Args=asdict(args))
    
if __name__ == "__main__":
    main()
