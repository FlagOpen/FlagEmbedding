"""
python hybrid_all_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw \
--dense_search_result_save_dir ./rerank_results/dense \
--sparse_search_result_save_dir ./rerank_results/sparse \
--colbert_search_result_save_dir ./rerank_results/colbert \
--hybrid_result_save_dir ./hybrid_search_results \
--top_k 200 \
--threads 32 \
--dense_weight 1 --sparse_weight 0.1 --colbert_weight 1
"""
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class EvalArgs:
    encoder: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of model'}
    )
    reranker: str = field(
        default='BAAI/bge-m3',
        metadata={'help': 'Name or path of reranker'}
    )
    languages: str = field(
        default="en",
        metadata={'help': 'Languages to evaluate. Avaliable languages: en ar fi ja ko ru es sv he th da de fr it nl pl pt hu vi ms km no tr zh_cn zh_hk zh_tw', 
                  "nargs": "+"}
    )
    top_k: int = field(
        default=200,
        metadata={'help': 'Use reranker to rerank top-k retrieval results'}
    )
    dense_weight: float = field(
        default=1,
        metadata={'help': 'Hybrid weight of sparse score'}
    )
    sparse_weight: float = field(
        default=0.3,
        metadata={'help': 'Hybrid weight of sparse score'}
    )
    colbert_weight: float = field(
        default=1,
        metadata={'help': 'Hybrid weight of sparse score'}
    )
    dense_search_result_save_dir: str = field(
        default='../rerank/unify_rerank_results/dense',
        metadata={'help': 'Dir to saving dense search results. Search results path is `dense_search_result_save_dir/{encoder}-{reranker}/{lang}.txt`'}
    )
    sparse_search_result_save_dir: str = field(
        default='../rerank/unify_rerank_results/sparse',
        metadata={'help': 'Dir to saving sparse search results. Search results path is `sparse_search_result_save_dir/{encoder}-{reranker}/{lang}.txt`'}
    )
    colbert_search_result_save_dir: str = field(
        default='../rerank/unify_rerank_results/colbert',
        metadata={'help': 'Dir to saving sparse search results. Search results path is `sparse_search_result_save_dir/{encoder}-{reranker}/{lang}.txt`'}
    )
    hybrid_result_save_dir: str = field(
        default='./search_results',
        metadata={'help': 'Dir to saving hybrid search results. Reranked results will be saved to `hybrid_result_save_dir/{encoder}-{reranker}/{lang}.txt`'}
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


def get_search_result_dict(search_result_path: str, top_k: int=1000):
    search_result_dict = {}
    flag = True
    for _, row in pd.read_csv(search_result_path, sep=' ', header=None).iterrows():
        qid = str(row.iloc[0])
        docid = row.iloc[2]
        rank = int(row.iloc[3])
        score = float(row.iloc[4])
        if qid not in search_result_dict:
            search_result_dict[qid] = []
            flag = False
        if rank > top_k:
            flag = True
        if flag:
            continue
        else:
            search_result_dict[qid].append((docid, score))
    return search_result_dict


def get_queries_dict(queries_path: str):
    queries_dict = {}
    for _, row in pd.read_csv(queries_path, sep='\t', header=None).iterrows():
        qid = str(row.iloc[0])
        query = row.iloc[1]
        queries_dict[qid] = query
    return queries_dict


def save_hybrid_results(sparse_search_result_dict: dict, dense_search_result_dict: dict, colbert_search_result_dict: dict, hybrid_result_save_path: str, top_k: int=1000, dense_weight: float=1, sparse_weight: float=0.3, colbert_weight: float=1):
    if not os.path.exists(os.path.dirname(hybrid_result_save_path)):
        os.makedirs(os.path.dirname(hybrid_result_save_path))
    
    qid_list = list(set(sparse_search_result_dict.keys()) | set(dense_search_result_dict.keys()) | set(colbert_search_result_dict.keys()))
    hybrid_results_list = []
    for qid in tqdm(qid_list, desc="Hybriding dense, sparse and colbert scores"):
        results = {}
        if qid in sparse_search_result_dict:
            for docid, score in sparse_search_result_dict[qid]:
                score = score / 0.3     # use 0.3 to restore
                results[docid] = score * sparse_weight
        if qid in dense_search_result_dict:
            for docid, score in dense_search_result_dict[qid]:
                if docid in results:
                    results[docid] = results[docid] + score * dense_weight
                else:
                    results[docid] = score * dense_weight
        if qid in colbert_search_result_dict:
            for docid, score in colbert_search_result_dict[qid]:
                if docid in results:
                    results[docid] = results[docid] + score * colbert_weight
                else:
                    results[docid] = score * colbert_weight
        hybrid_results = [(docid, score) for docid, score in results.items()]
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        hybrid_results_list.append(hybrid_results[:top_k])
    
    with open(hybrid_result_save_path, 'w', encoding='utf-8') as f:
        for qid, hybrid_results in tqdm(zip(qid_list, hybrid_results_list), desc="Saving hybrid search results"):
            for rank, docid_score in enumerate(hybrid_results):
                docid, score = docid_score
                line = f"{qid} Q0 {docid} {rank+1} {score:.6f} Faiss-&-Anserini"
                f.write(line + '\n')


def main():
    parser = HfArgumentParser([EvalArgs])
    eval_args = parser.parse_args_into_dataclasses()[0]
    eval_args: EvalArgs
    
    languages = check_languages(eval_args.languages)
    
    if os.path.basename(eval_args.encoder).startswith('checkpoint-'):
        eval_args.encoder = os.path.dirname(eval_args.encoder) + '_' + os.path.basename(eval_args.encoder)
    
    if os.path.basename(eval_args.reranker).startswith('checkpoint-'):
        eval_args.reranker = os.path.dirname(eval_args.reranker) + '_' + os.path.basename(eval_args.reranker)
    
    dir_name = f"{os.path.basename(eval_args.encoder)}-{os.path.basename(eval_args.reranker)}"
    
    if eval_args.threads > 1:
        threads = min(len(languages), eval_args.threads)
        pool = multiprocessing.Pool(processes=threads)
        for lang in languages:
            print("**************************************************")
            print(f"Start hybrid search results of {lang} ...")
            
            hybrid_result_save_path = os.path.join(eval_args.hybrid_result_save_dir, dir_name, f"{lang}.txt")
            
            sparse_search_result_save_dir = os.path.join(eval_args.sparse_search_result_save_dir, dir_name)
            
            sparse_search_result_path = os.path.join(sparse_search_result_save_dir, f"{lang}.txt")
            
            sparse_search_result_dict = get_search_result_dict(sparse_search_result_path, top_k=eval_args.top_k)
            
            dense_search_result_save_dir = os.path.join(eval_args.dense_search_result_save_dir, dir_name)
            
            dense_search_result_path = os.path.join(dense_search_result_save_dir, f"{lang}.txt")
            
            dense_search_result_dict = get_search_result_dict(dense_search_result_path, top_k=eval_args.top_k)
            
            colbert_search_result_save_dir = os.path.join(eval_args.colbert_search_result_save_dir, dir_name)
            
            colbert_search_result_path = os.path.join(colbert_search_result_save_dir, f"{lang}.txt")
            
            colbert_search_result_dict = get_search_result_dict(colbert_search_result_path, top_k=eval_args.top_k)
            
            pool.apply_async(save_hybrid_results, args=(
                sparse_search_result_dict,
                dense_search_result_dict,
                colbert_search_result_dict,
                hybrid_result_save_path,
                eval_args.top_k,
                eval_args.dense_weight,
                eval_args.sparse_weight,
                eval_args.colbert_weight
            ))
        pool.close()
        pool.join()
    else:
        for lang in languages:
            print("**************************************************")
            print(f"Start hybrid search results of {lang} ...")
            
            hybrid_result_save_path = os.path.join(eval_args.hybrid_result_save_dir, dir_name, f"{lang}.txt")
            
            sparse_search_result_save_dir = os.path.join(eval_args.sparse_search_result_save_dir, dir_name)
            
            sparse_search_result_path = os.path.join(sparse_search_result_save_dir, f"{lang}.txt")
            
            sparse_search_result_dict = get_search_result_dict(sparse_search_result_path, top_k=eval_args.top_k)
            
            dense_search_result_save_dir = os.path.join(eval_args.dense_search_result_save_dir, dir_name)
            
            dense_search_result_path = os.path.join(dense_search_result_save_dir, f"{lang}.txt")
            
            dense_search_result_dict = get_search_result_dict(dense_search_result_path, top_k=eval_args.top_k)
            
            colbert_search_result_save_dir = os.path.join(eval_args.colbert_search_result_save_dir, dir_name)
            
            colbert_search_result_path = os.path.join(colbert_search_result_save_dir, f"{lang}.txt")
            
            colbert_search_result_dict = get_search_result_dict(colbert_search_result_path, top_k=eval_args.top_k)
            
            save_hybrid_results(
                sparse_search_result_dict=sparse_search_result_dict, 
                dense_search_result_dict=dense_search_result_dict, 
                colbert_search_result_dict=colbert_search_result_dict,
                hybrid_result_save_path=hybrid_result_save_path,
                top_k=eval_args.top_k,
                dense_weight=eval_args.dense_weight,
                sparse_weight=eval_args.sparse_weight,
                colbert_weight=eval_args.colbert_weight
            )
    
    print("==================================================")
    print("Finish generating reranked results with following model and reranker:")
    print(eval_args.encoder)
    print(eval_args.reranker)


if __name__ == "__main__":
    main()
