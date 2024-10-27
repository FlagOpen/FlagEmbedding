"""
# 1. Rerank Search Results
python step0-rerank_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ../dense_retrieval/search_results \
--rerank_result_save_dir ./rerank_results \
--top_k 200 \
--batch_size 4 \
--max_query_length 512 \
--max_passage_length 8192 \
--pooling_method cls \
--normalize_embeddings True \
--dense_weight 0.15 --sparse_weight 0.5 --colbert_weight 0.35 \
--num_shards 1 --shard_id 0 --cuda_id 0

# 2. Print and Save Evaluation Results
python step1-eval_rerank_mldr.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./rerank_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10
"""
import os
import json
import platform
import subprocess
import numpy as np
from pprint import pprint
from collections import defaultdict
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
    reranker: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of reranker'}
    )
    encoder: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of encoder'}
    )
    search_result_save_dir: str = field(
        default='./rerank_results',
        metadata={'help': 'Dir to saving search results. Search results path is `result_save_dir/{encoder}-{reranker}/{lang}.txt`'}
    )
    qrels_dir: str = field(
        default='../qrels',
        metadata={'help': 'Dir to topics and qrels.'}
    )
    metrics: str = field(
        default="ndcg@10",
        metadata={'help': 'Metrics to evaluate. Avaliable metrics: ndcg@k, recall@k', 
                  "nargs": "+"}
    )
    eval_result_save_dir: str = field(
        default='./reranker_evaluation_results',
        metadata={'help': 'Dir to saving evaluation results. Evaluation results will be saved to `eval_result_save_dir/{encoder}-{reranker}.json`'}
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


def save_results(model_name: str, reranker_name: str, results: dict, save_path: str, eval_languages: list):
    try:
        results['average'] = compute_average(results)
    except:
        results['average'] = None
        pass
    pprint(results)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    results_dict = {
        'reranker': reranker_name,
        'model': model_name,
        'results': results
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    print(f'Results of evaluating `{reranker_name}` on `{eval_languages}` based on `{model_name}` saved at `{save_path}`')


def map_metric(metric: str):
    metric, k = metric.split('@')
    if metric.lower() == 'ndcg':
        return k, f'ndcg_cut.{k}'
    elif metric.lower() == 'recall':
        return k, f'recall.{k}'
    else:
        raise ValueError(f"Unkown metric: {metric}")


def evaluate(script_path: str, qrels_path, search_result_path, metrics: list):
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


def merge_search_result(search_result_save_dir: str, lang: str):
    lang_files = [file for file in os.listdir(search_result_save_dir) if f'{lang}_' in file]
    shard_info_dict = defaultdict(set)
    for file in lang_files:
        file_name = file.split('.')[0]
        shard_info = file_name.split('_')[1]
        shard_id, num_shards = int(shard_info.split('-')[0]), int(shard_info.split('-')[2])
        assert shard_id < num_shards
        shard_info_dict[num_shards].add(shard_id)
    flag = False
    for num_shards, shard_ids in shard_info_dict.items():
        if len(shard_ids) != num_shards:
            flag = False
        else:
            flag = True
            lang_paths = os.path.join(search_result_save_dir, f'{lang}_*-of-{num_shards}.txt')
            save_path = os.path.join(search_result_save_dir, f'{lang}.txt')
            cmd = f'cat {lang_paths} > {save_path}'
            os.system(cmd)
            break
    if not flag:
        raise ValueError(f"Fail to find complete search results of {lang} in {search_result_save_dir}")


def main():
    parser = HfArgumentParser([EvalArgs])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs
    
    script_path = download_evaluation_script('trec_eval')
    
    languages = check_languages(eval_args.languages)
    
    if 'checkpoint-' in os.path.basename(eval_args.encoder):
        eval_args.encoder = os.path.dirname(eval_args.encoder) + '_' + os.path.basename(eval_args.encoder)
    
    if 'checkpoint-' in os.path.basename(eval_args.reranker):
        eval_args.reranker = os.path.dirname(eval_args.reranker) + '_' + os.path.basename(eval_args.reranker)
    
    try:
        for sub_dir in ['colbert', 'sparse', 'dense', 'colbert+sparse+dense']:
            results = {}
            for lang in languages:
                qrels_path = os.path.join(eval_args.qrels_dir, f"qrels.mldr-v1.0-{lang}-test.tsv")
                
                search_result_save_dir = os.path.join(eval_args.search_result_save_dir, sub_dir, f"{os.path.basename(eval_args.encoder)}-{os.path.basename(eval_args.reranker)}")
                search_result_path = os.path.join(search_result_save_dir, f"{lang}.txt")
                if not os.path.exists(search_result_path):
                    merge_search_result(search_result_save_dir, lang)
                    assert os.path.exists(search_result_path)
                
                result = evaluate(script_path, qrels_path, search_result_path, eval_args.metrics)
                results[lang] = result
            
            print("****************************")
            print(sub_dir + ":")
            save_results(
                model_name=eval_args.encoder,
                reranker_name=eval_args.reranker,
                results=results,
                save_path=os.path.join(eval_args.eval_result_save_dir, sub_dir, f"{os.path.basename(eval_args.encoder)}-{os.path.basename(eval_args.reranker)}.json"),
                eval_languages=languages,
            )
    except:
        results = {}
        for lang in languages:
            qrels_path = os.path.join(eval_args.qrels_dir, f"qrels.mldr-v1.0-{lang}-test.tsv")
            
            search_result_save_dir = os.path.join(eval_args.search_result_save_dir, f"{os.path.basename(eval_args.encoder)}-{os.path.basename(eval_args.reranker)}")
            search_result_path = os.path.join(search_result_save_dir, f"{lang}.txt")
            if not os.path.exists(search_result_path):
                merge_search_result(search_result_save_dir, lang)
                assert os.path.exists(search_result_path)
            
            result = evaluate(script_path, qrels_path, search_result_path, eval_args.metrics)
            results[lang] = result
        save_results(
            model_name=eval_args.encoder,
            reranker_name=eval_args.reranker,
            results=results,
            save_path=os.path.join(eval_args.eval_result_save_dir, f"{os.path.basename(eval_args.encoder)}-{os.path.basename(eval_args.reranker)}.json"),
            eval_languages=languages,
        )
    
    print("==================================================")
    print("Finish generating evaluation results with following model and reranker:")
    print(eval_args.encoder)
    print(eval_args.reranker)

if __name__ == "__main__":
    main()
