"""
# 1. Search Dense and Sparse Results
../dense_retrieval
../sparse_retrieval

# 2. Hybrid Dense and Sparse Search Results
python step0-hybrid_search_results.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--dense_search_result_save_dir ../dense_retrieval/search_results \
--sparse_search_result_save_dir ../sparse_retrieval/search_results \
--hybrid_result_save_dir ./search_results \
--top_k 1000 \
--dense_weight 0.2 --sparse_weight 0.8

# 3. Print and Save Evaluation Results
python step1-eval_hybrid_mldr.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True
"""
import os
import json
import platform
import subprocess
import numpy as np
from pprint import pprint
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from pyserini.util import download_evaluation_script


@dataclass
class EvalArgs:
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: ar de en es fr hi it ja ko pt ru th zh', 
                  "nargs": "+"}
    )
    model_name_or_path: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of model'}
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
        default='./output_results',
        metadata={'help': 'Dir to saving search results. Search results path is `result_save_dir/{model_name_or_path}/{lang}.txt`'}
    )
    qrels_dir: str = field(
        default='../qrels',
        metadata={'help': 'Dir to qrels.'}
    )
    metrics: str = field(
        default="ndcg@10",
        metadata={'help': 'Metrics to evaluate. Avaliable metrics: ndcg@k, recall@k', 
                  "nargs": "+"}
    )
    eval_result_save_dir: str = field(
        default='./eval_results',
        metadata={'help': 'Dir to saving evaluation results. Evaluation results will be saved to `eval_result_save_dir/{model_name_or_path}.json`'}
    )


def check_languages(languages):
    if isinstance(languages, str):
        languages = [languages]
    avaliable_languages = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']
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


def map_metric(metric: str):
    metric, k = metric.split('@')
    if metric.lower() == 'ndcg':
        return k, f'ndcg_cut.{k}'
    elif metric.lower() == 'recall':
        return k, f'recall.{k}'
    else:
        raise ValueError(f"Unkown metric: {metric}")


def evaluate(script_path, qrels_path, search_result_path, metrics: list):
    cmd_prefix = ['java', '-jar', script_path]
    
    results = {}
    for metric in metrics:
        k, mapped_metric = map_metric(metric)
        args = ['-c', '-M', str(k), '-m', mapped_metric, qrels_path, search_result_path]
        cmd = cmd_prefix + args
        
        # print(f'Running command: {cmd}')
        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=shell)
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        result_str = stdout.decode("utf-8")
        try:
            results[metric] = float(result_str.split(' ')[-1].split('\t')[-1])
        except:
            results[metric] = result_str
    return results


def main():
    parser = HfArgumentParser([EvalArgs])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    script_path = download_evaluation_script('trec_eval')
    
    if eval_args.model_name_or_path[-1] == '/':
        eval_args.model_name_or_path = eval_args.model_name_or_path[:-1]
    if os.path.basename(eval_args.model_name_or_path).startswith('checkpoint-'):
        eval_args.model_name_or_path = os.path.dirname(eval_args.model_name_or_path) + '_' + os.path.basename(eval_args.model_name_or_path)
    
    results = {}
    for lang in languages:
        qrels_path = os.path.join(eval_args.qrels_dir, f"qrels.mldr-v1.0-{lang}-test.tsv")
        
        search_result_save_dir = os.path.join(eval_args.search_result_save_dir, os.path.basename(eval_args.model_name_or_path))
        search_result_path = os.path.join(search_result_save_dir, f"{lang}.txt")
        
        result = evaluate(script_path, qrels_path, search_result_path, eval_args.metrics)
        results[lang] = result
    
    save_results(
        model_name=eval_args.model_name_or_path,
        pooling_method=eval_args.pooling_method,
        normalize_embeddings=eval_args.normalize_embeddings,
        results=results,
        save_path=os.path.join(eval_args.eval_result_save_dir, f"{os.path.basename(eval_args.model_name_or_path)}.json"),
        eval_languages=languages
    )
    print("==================================================")
    print("Finish generating evaluation results with following model:")
    print(eval_args.model_name_or_path)


if __name__ == "__main__":
    main()
