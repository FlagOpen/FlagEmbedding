"""
# Ref: https://github.com/texttron/tevatron/tree/main/examples/unicoil
# 1. Generate Query and Corpus Sparse Vector
python step0-encode_query-and-corpus.py \
--encoder BAAI/bge-m3 \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--qa_data_dir ../qa_data \
--save_dir ./encoded_query-and-corpus \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 1024 \
--pooling_method cls \
--normalize_embeddings True

# 2. Output Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--encoded_query_and_corpus_save_dir ./encoded_query-and-corpus \
--result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--threads 16 \
--hits 1000

# 3. Print and Save Evaluation Results
python step2-eval_sparse_mkqa.py \
--encoder BAAI/bge-m3 \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32 \
--pooling_method cls \
--normalize_embeddings True
"""
import os
import sys
import json
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser

sys.path.append("..")

from utils.normalize_text import normalize
from utils.evaluation import evaluate_recall_qa


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: en ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw', 
                  "nargs": "+"}
    )
    encoder: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of encoder'}
    )
    pooling_method: str = field(
        default='cls',
        metadata={'help': "Pooling method. Avaliable methods: 'cls', 'mean'"}
    )
    normalize_embeddings: bool = field(
        default=True,
        metadata={'help': "Normalize embeddings or not"}
    )
    search_result_save_dir: str = field(
        default='./search_results',
        metadata={'help': 'Dir to saving search results. Search results path is `result_save_dir/{encoder}/{lang}.txt`'}
    )
    qa_data_dir: str = field(
        default='../qa_data',
        metadata={'help': 'Dir to qa data.'}
    )
    metrics: str = field(
        default="recall@20",
        metadata={'help': 'Metrics to evaluate. Avaliable metrics: recall@k', 
                  "nargs": "+"}
    )
    eval_result_save_dir: str = field(
        default='./eval_results',
        metadata={'help': 'Dir to saving evaluation results. Evaluation results will be saved to `eval_result_save_dir/{encoder}.json`'}
    )
    threads: int = field(
        default=1,
        metadata={"help": "num of evaluation threads. <= 1 means single thread"}
    )


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['en', 'ar', 'fi', 'ja', 'ko', 'ru', 'es', 'sv', 'he', 'th', 'da', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'hu', 'vi', 'ms', 'km', 'no', 'tr', 'zh_cn', 'zh_hk', 'zh_tw']
    for lang in languages:
        if lang not in avaliable_languages:
            raise ValueError(f"Language `{lang}` is not supported. Avaliable languages: {avaliable_languages}")
    return languages


def compute_average(results: dict):
    average_results = {}
    for _, result in results.items():
        for metric, score in result.items():
            if metric not in average_results:
                average_results[metric] = []
            average_results[metric].append(score)
    for metric, scores in average_results.items():
        average_results[metric] = np.mean(scores)
    return average_results


def save_results(model_name: str, pooling_method: str, normalize_embeddings: bool, results: dict, save_path: str, eval_languages: list):
    try:
        results['average'] = compute_average(results)
    except:
        results['average'] = None
        pass
    pprint(results)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    results_dict = {
        'model': model_name,
        'pooling_method': pooling_method,
        'normalize_embeddings': normalize_embeddings,
        'results': results
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    print(f'Results of evaluating `{model_name}` on `{eval_languages}` saved at `{save_path}`')


def get_corpus_dict():
    corpus_dict = {}
    corpus = datasets.load_dataset('BeIR/nq', 'corpus')['corpus']
    for data in tqdm(corpus, desc="Loading corpus"):
        _id = str(data['_id'])
        content = f"{data['title']}\n{data['text']}".lower()
        content = normalize(content)
        corpus_dict[_id] = content
    return corpus_dict


def get_qa_dict(qa_path: str):
    qa_dict = {}
    dataset = datasets.load_dataset('json', data_files=qa_path)['train']
    for data in dataset:
        qid = str(data['id'])
        answers = data['answers']
        qa_dict[qid] = answers
    return qa_dict


def get_search_result_dict(search_result_path: str, top_k: int=100):
    search_result_dict = {}
    flag = True
    for _, row in pd.read_csv(search_result_path, sep=' ', header=None).iterrows():
        qid = str(row.iloc[0])
        docid = str(row.iloc[2])
        rank = int(row.iloc[3])
        if qid not in search_result_dict:
            search_result_dict[qid] = []
            flag = False
        if rank > top_k:
            flag = True
        if flag:
            continue
        else:
            search_result_dict[qid].append(docid)
    return search_result_dict


def evaluate(corpus_dict: dict, qa_dict: dict, search_result_path: str, metrics: list):
    top_k = max([int(metric.split('@')[-1]) for metric in metrics])
    search_result_dict = get_search_result_dict(search_result_path, top_k=int(top_k))
    
    search_results = []
    ground_truths = []
    for qid, docid_list in search_result_dict.items():
        answers = qa_dict[qid]
        doc_list = [corpus_dict[docid] for docid in docid_list]
        search_results.append(doc_list)
        ground_truths.append(answers)
    
    results = {}
    metrics = sorted([metric.lower() for metric in metrics])
    for metric in metrics:
        metric, k = metric.split('@')
        k = int(k)
        assert metric in ['recall'], f"Metric `{metric}` is not supported."
        if metric == 'recall':
            results[f'Recall@{k}'] = evaluate_recall_qa(search_results, ground_truths, k=k)
    return results


def main():
    parser = HfArgumentParser([EvalArgs])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs
    
    corpus_dict = get_corpus_dict()
    
    languages = check_languages(eval_args.languages)
    
    if eval_args.encoder[-1] == '/':
        eval_args.encoder = eval_args.encoder[:-1]
    
    if os.path.basename(eval_args.encoder).startswith('checkpoint-'):
        eval_args.encoder = os.path.dirname(eval_args.encoder) + '_' + os.path.basename(eval_args.encoder)
    
    results = {}
    if eval_args.threads > 1:
        threads = min(len(languages), eval_args.threads)
        pool = multiprocessing.Pool(processes=threads)
        results_list = []
        for lang in languages:
            print("*****************************")
            print(f"Start evaluating {lang} ...")
            qa_path = os.path.join(eval_args.qa_data_dir, f"{lang}.jsonl")
            qa_dict = get_qa_dict(qa_path)
            
            search_result_save_dir = os.path.join(eval_args.search_result_save_dir, os.path.basename(eval_args.encoder))
            search_result_path = os.path.join(search_result_save_dir, f"{lang}.txt")
            
            results_list.append(pool.apply_async(evaluate, args=(corpus_dict, qa_dict, search_result_path, eval_args.metrics)))
        pool.close()
        pool.join()
        for i, lang in enumerate(languages):
            results[lang] = results_list[i].get()
    else:
        for lang in languages:
            print("*****************************")
            print(f"Start evaluating {lang} ...")
            qa_path = os.path.join(eval_args.qa_data_dir, f"{lang}.jsonl")
            qa_dict = get_qa_dict(qa_path)
            
            search_result_save_dir = os.path.join(eval_args.search_result_save_dir, os.path.basename(eval_args.encoder))
            search_result_path = os.path.join(search_result_save_dir, f"{lang}.txt")
            
            result = evaluate(corpus_dict, qa_dict, search_result_path, eval_args.metrics)
            results[lang] = result
    
    save_results(
        model_name=eval_args.encoder,
        pooling_method=eval_args.pooling_method,
        normalize_embeddings=eval_args.normalize_embeddings,
        results=results,
        save_path=os.path.join(eval_args.eval_result_save_dir, f"{os.path.basename(eval_args.encoder)}.json"),
        eval_languages=languages
    )
    print("==================================================")
    print("Finish generating evaluation results with following model:")
    print(eval_args.encoder)


if __name__ == "__main__":
    main()
