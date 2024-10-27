import json
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytrec_eval
from transformers import HfArgumentParser
from FlagEmbedding import FlagReranker

@dataclass
class Args():
    input_path: str = field(
        default="",
        metadata={'help': """
        The data path points to a file in JSONL format.
        Each line contains `query`, `pos`, and `neg`. Here, `query` is a string (`str`), 
        while both `pos` and `neg` are lists of strings (`List[str]`).
        If each line includes `pos_label_scores`, it will use to compute `ndcg@k`, else it will set default `1`.
        """}
    )
    metrics: List[str] = field(
        default=None, # usage example: recall mrr ndcg
        metadata={'help': 'The evaluation metrics, you can set recall / mrr / ndcg'}
    )
    k_values: List[int] = field(
        default=None,
        metadata={'help': 'Present the top-k metrics evaluation.'}
    )
    cache_dir: str = field(
        default=None,
        metadata={'help': 'The path to store the cache of reranker.'}
    )
    use_fp16: bool = field(
        default=True,
        metadata={'help': 'Whether to use fp16 to accelerate inference, it is not suitable for CPU only inference.'}
    )
    batch_size: int = field(
        default=512
    )
    max_length: int = field(
        default=1024
    )


def evaluate_mrr(predicts, labels, cutoffs):
    """
    Evaluate MRR.
    """
    metrics = {}

    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(predicts, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(predicts)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    return metrics

def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    input_path = args.input_path
    metrics = args.metrics if args.metrics is not None else ['recall', 'mrr', 'ndcg', 'map', 'precision']
    k_values = args.k_values if args.k_values is not None else [1, 5, 10, 50, 100]
    cache_dir = args.cache_dir
    use_fp16 = args.use_fp16
    batch_size = args.batch_size
    max_length = args.max_length

    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', cache_dir=cache_dir, use_fp16=use_fp16)

    data = []
    data_num = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))

    pairs = []
    for d in data:
        data_num.append(0)
        passages = []
        passages.extend(d['pos'])
        passages.extend(d['neg'])
        for p in passages:
            pairs.append((d['query'], p))
            data_num[-1] += 1

    scores = reranker.compute_score(pairs, batch_size=batch_size, max_length=max_length)
    scores = np.asarray(scores)
    scores = scores.reshape(-1)

    start_num = 0
    ground_truths = {}
    labels = []
    for i in range(len(data)):
        tmp = {}
        tmp_labels = []
        for ind in range(len(data[i]['pos'])):
            try:
                tmp[str(start_num + ind)] = int(data[i]['pos_label_scores'][ind])
            except Exception as e:
                # print(e)
                tmp[str(start_num + ind)] = 1
            tmp_labels.append(start_num + ind)
        ground_truths[str(i)] = tmp
        start_num += data_num[i]
        labels.append(tmp_labels)

    start_num = 0
    rerank_results = {}
    predicts = []
    for i in range(len(data)):
        tmp = {}
        tmp_predicts = [(start_num + ind, scores[start_num + ind]) for ind in range(data_num[i])]
        tmp_predicts = [idx for (idx, _) in sorted(tmp_predicts, key=lambda x: x[1], reverse=True)]
        for ind in range(data_num[i]):
            tmp[str(start_num + ind)] = float(scores[start_num + ind])
        rerank_results[str(i)] = tmp
        start_num += data_num[i]
        predicts.append(tmp_predicts)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"Precision@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(ground_truths,
                                               {map_string, ndcg_string, recall_string, precision_string})

    scores = evaluator.evaluate(rerank_results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"Precision@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"Precision@{k}"] = round(precision[f"Precision@{k}"] / len(scores), 5)

    mrr = evaluate_mrr(predicts, labels, k_values)

    if 'mrr' in metrics:
        print(mrr)
    if 'recall' in metrics:
        print(recall)
    if 'ndcg' in metrics:
        print(ndcg)
    if 'map' in metrics:
        print(_map)
    if 'precision' in metrics:
        print(precision)

if __name__ == "__main__":
    main()