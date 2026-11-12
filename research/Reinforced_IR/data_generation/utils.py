import re
import random
import os

import faiss
import numpy as np
import pytrec_eval
import torch
import gc

from transformers import AutoModel
from tqdm import trange, tqdm
from typing import List, Dict, Tuple, Union

from agent import GPTAgent, LLMAgent, LLMInstructAgent

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    numbers = [int(num) for num in numbers]
    return numbers

def get_distill_data(
        llm_for_rank = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        train_data: List = None,
        prompts: List[str] = None,
):
    generated_rank_results = llm_for_rank.generate(
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    for d, res in zip(train_data, generated_rank_results):
        res = extract_numbers(res)
        passages = []
        passages.extend(d['pos'])
        passages.extend(d['neg'])
        if 0 not in res and len(passages) in res:
            res = [e - 1 for e in res]
        except_res = [i for i in res if i >= len(passages)]
        for e in except_res:
            res.remove(e)
        if len(res) < len(passages):
            print(res)
            for i in range(len(passages)):
                if i not in res:
                    res.append(i)

        d['pos'] = []
        d['neg'] = []
        for i in res[:1]:
            d['pos'].append(passages[i])
        for i in res[1:]:
            d['neg'].append(passages[i])
        d['pos_scores'] = [1]
        d['neg_scores'] = [1 / (i + 1) for i in range(len(res) - 1)]

    return train_data


def generate_bge_train_data(
    retrieval_model,
    batch_size: int = 512,
    max_length: int = 512,
    queries_corpus: Union[List[dict], List[List[dict]]] = None,
    dtype: str = 'passage',
    corpus: List[str] = None,
    filter_data: bool = False,
    filter_num: int = 20,
    emb_save_path: str = None,
    ignore_prefix: bool = False,
    neg_type: str = 'hard'
):
    if corpus is None:
        corpus = [d[dtype] for d in queries_corpus]
    queries = [d['query'] for d in queries_corpus]
    answers = [d['answer'] for d in queries_corpus]

    queries_emb = retrieval_model.encode_queries(queries, batch_size=batch_size,
                                                 max_length=max_length)
    # * 0.8 + retrieval_model.encode_corpus(answers, batch_size=batch_size, max_length=max_length) * 0.2
    answers_emb = retrieval_model.encode_corpus(answers, batch_size=batch_size, max_length=max_length)
    if emb_save_path is not None:
        if os.path.exists(emb_save_path):
            if ignore_prefix:
                doc_emb = np.vstack(
                    (
                        retrieval_model.encode_corpus(corpus[: len(queries_emb)], batch_size=batch_size,
                                                      max_length=max_length),
                        np.load(emb_save_path)
                    )
                )
            else:
                doc_emb = np.load(emb_save_path)
        else:
            doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
            try:
                os.makedirs('/'.join(emb_save_path.split('/')[:-1]), exist_ok=True)
            except:
                pass
            if ignore_prefix:
                np.save(emb_save_path, doc_emb[len(queries_emb):])
            else:
                np.save(emb_save_path, doc_emb)
    else:
        doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)

    print('len doc emb:', len(doc_emb))

    all_scores, all_indices = search(queries_emb, doc_emb, 2000)
    _, all_answers_indices = search(answers_emb, doc_emb, 2000)

    train_data = []

    find_idxs = []
    for i in range(len(all_indices)):
        if i in list(all_indices[i]):
            find_idxs.append(list(all_indices[i]).index(i))
        else:
            find_idxs.append(-1)
    print(find_idxs)

    answers_find_idxs = []
    for i in range(len(all_answers_indices)):
        if i in list(all_answers_indices[i]):
            answers_find_idxs.append(list(all_answers_indices[i]).index(i))
        else:
            answers_find_idxs.append(-1)

    for i in trange(len(queries), desc='generate train set'):

        if find_idxs[i] == -1:  # remove false pairs
            # continue
            # neg_ids = random.sample(list(range(len(corpus))), k=50)
            neg_ids = random.sample(list(all_indices[i][30:200]), k=50)
        else:
            uses_idx = -1
            for j in range(find_idxs[i] + 1, 2000):
                if all_scores[i][j] <= all_scores[i][find_idxs[i]] * 0.95:
                    uses_idx = j
                    break
            if uses_idx == -1:
                # continue
                # neg_ids = random.sample(list(range(len(corpus))), k=50)
                neg_ids = random.sample(list(all_indices[i][30:200]), k=50)
            else:
                neg_ids = list(all_indices[i][uses_idx: uses_idx + 50])
        # neg_ids = list(all_indices[i][:50])
        if neg_type == 'random':
            neg_ids = random.sample(list(range(len(corpus))), k=50)
        elif neg_type == 'hard':
            # neg_ids = list(all_indices[i][:50])
            neg_ids = random.sample(list(all_indices[i][30:200]), k=50)
            tmp_ids = [(e, list(all_indices[i]).index(e)) for e in neg_ids]
            tmp_ids = sorted(tmp_ids, key=lambda x: x[1])
            neg_ids = [e[0] for e in tmp_ids]
        else:
            tmp_ids = [(e, list(all_indices[i]).index(e)) for e in neg_ids]
            tmp_ids = sorted(tmp_ids, key=lambda x: x[1])
            neg_ids = [e[0] for e in tmp_ids]

        if answers_find_idxs[i] == -1:  # remove false pairs
            # continue
            neg_answers_ids = random.sample(list(range(len(corpus))), k=50)
        else:
            uses_idx = -1
            for j in range(answers_find_idxs[i] + 1, 2000):
                if all_scores[i][j] <= all_scores[i][answers_find_idxs[i]] * 0.95:
                    uses_idx = j
                    break
            if uses_idx == -1:
                # continue
                neg_answers_ids = random.sample(list(range(len(corpus))), k=50)
            else:
                neg_answers_ids = list(all_answers_indices[i][uses_idx: uses_idx + 50])

        query = queries[i]
        answer = answers[i]
        pos = [corpus[i]]
        negs = [corpus[j] for j in neg_ids]
        while pos[0] in negs:
            negs.remove(pos[0])
        new_negs = []
        for e in negs:
            if e not in new_negs and len(new_negs) < 15:
                new_negs.append(e)
        negs = new_negs

        negs_answer = [corpus[j] for j in neg_answers_ids]
        while pos[0] in negs_answer:
            negs_answer.remove(pos[0])
        new_negs_answer = []
        for e in negs_answer:
            if e not in new_negs_answer and len(new_negs_answer) < 15:
                new_negs_answer.append(e)
        negs_answer = new_negs_answer

        train_data.append(
            {
                'query': query,
                'answer': answer,
                'pos': pos,
                'neg': negs,
                'neg_answer': negs_answer
            }
        )

    if filter_data:
        print(filter_data)
        new_train_data = []
        for i in range(len(all_indices)):
            if i in list(all_indices[i]):
                seached_idx = list(all_indices[i]).index(i)
            else:
                seached_idx = len(all_indices) + 999
            if seached_idx < filter_num:
                new_train_data.append(train_data[i])
        train_data = new_train_data

    print(len(train_data))

    return train_data


def generate_llm_dpo_train_data(
    queries_corpus_list: List[List[dict]] = None,
    search_dtype: str = 'answer',
    result_dtype: str = 'passage',
    retrieval_model: AutoModel = None,
    threshold: float = 0.95,
    batch_size: int = 512,
    max_length: int = 1024,
    use_rule1: bool = True
):
    data = []

    queries_list = []
    corpus = []
    raw_queries = []
    for qc in queries_corpus_list:
        raw_queries = [d['query'] for d in qc]
        if 'new_query' in qc[0].keys():
            queries_list.append([d['new_query'] for d in qc])
        else:
            queries_list.append([d[search_dtype] for d in qc])
        corpus = [d[result_dtype] for d in qc]

    doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
    raw_queries_emb = retrieval_model.encode_queries(raw_queries, batch_size=batch_size, max_length=max_length)
    raw_scores = np.einsum('ij,ij->i', raw_queries_emb, doc_emb)
    all_scores_list = []
    for queries in queries_list:
        # queries = ['Generate the topic about this passage: ' + q for q in queries]
        queries_emb = raw_queries_emb * 0.8 + retrieval_model.encode_queries(queries, batch_size=batch_size,
                                                                             max_length=max_length) * 0.2
        # queries_emb = raw_queries_emb
        all_scores_list.append(np.einsum('ij,ij->i', queries_emb, doc_emb))

    for i in range(len(all_scores_list[0])):
        raw_score = raw_scores[i]
        all_scores = [e[i] for e in all_scores_list]
        items = [(idx, all_scores[idx]) for idx in range(len(all_scores))]
        sorted_idx = [idx for idx, _ in sorted(items, key=lambda x: x[1], reverse=False)]
        min_score = max(all_scores)
        for idx in sorted_idx:
            if abs(1 - all_scores[idx] / raw_score) < 0.1:
                min_score = all_scores[idx]
                break
        min_score = min(all_scores)
        max_score = max(all_scores)

        if use_rule1:
            if max_score > raw_score and (max_score - raw_score * 0.8) * threshold >= (min_score - raw_score * 0.8):
                # print('use')
                tmp = {
                    'prompt': queries_corpus_list[0][i]['query'],
                    'chosen': queries_corpus_list[all_scores.index(max_score)][i][search_dtype],
                    'rejected': queries_corpus_list[all_scores.index(min_score)][i][search_dtype],
                }
                tmp['chosen_score'] = float(max_score / raw_score)
                tmp['rejected_score'] = float(min_score / raw_score)
                data.append(tmp)
        else:
            if (max_score - raw_score * 0.8) * threshold >= (min_score - raw_score * 0.8):
                # print('use')
                tmp = {
                    'prompt': queries_corpus_list[0][i]['query'],
                    'chosen': queries_corpus_list[all_scores.index(max_score)][i][search_dtype],
                    'rejected': queries_corpus_list[all_scores.index(min_score)][i][search_dtype],
                }
                tmp['chosen_score'] = float(max_score / raw_score)
                tmp['rejected_score'] = float(min_score / raw_score)
                data.append(tmp)

    return data


def evaluate_mrr(qrels: Dict[str, Dict[str, int]],
                 results: Dict[str, Dict[str, float]],
                 k_values: List[int]) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)

    return MRR


def search(queries_emb, doc_emb, topk: int = 100):
    gc.collect()
    torch.cuda.empty_cache()

    faiss_index = faiss.index_factory(doc_emb.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    doc_emb = doc_emb.astype(np.float32)
    faiss_index.train(doc_emb)
    faiss_index.add(doc_emb)

    dev_query_size = queries_emb.shape[0]
    all_scores = []
    all_indices = []
    for i in tqdm(range(0, dev_query_size, 32), desc="Searching"):
        j = min(i + 32, dev_query_size)
        query_embedding = queries_emb[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=topk)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    return all_scores, all_indices


def evaluate(metrics: List[str] = ['recall', 'mrr', 'ndcg'],
             k_values: List[int] = [1, 10],
             ground_truths: List[Dict] = None,
             predicts: List = None,
             scores: List = None):

    retrieval_results = {}
    for i in range(len(predicts)):
        tmp = {}
        for j in range(len(predicts[0])):
            tmp[str(predicts[i][j])] = float(scores[i][j])
        retrieval_results[str(i)] = tmp

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

    scores = evaluator.evaluate(retrieval_results)

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

    mrr = evaluate_mrr(ground_truths, retrieval_results, k_values)

    data = {}

    if 'mrr' in metrics:
        data['mrr'] = mrr
    if 'recall' in metrics:
        data['recall'] = recall
    if 'ndcg' in metrics:
        data['ndcg'] = ndcg
    if 'map' in metrics:
        data['map'] = _map
    if 'precision' in metrics:
        data['precision'] = precision

    return data


def evaluate_better(metrics: List[str] = ['recall', 'mrr', 'ndcg'],
                    k_values: List[int] = [1, 10],
                    ground_truths: List[Dict] = None,
                    retrieval_results: List[Dict] = None):
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

    scores = evaluator.evaluate(retrieval_results)

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

    mrr = evaluate_mrr(ground_truths, retrieval_results, k_values)

    data = {}

    if 'mrr' in metrics:
        data['mrr'] = mrr
    if 'recall' in metrics:
        data['recall'] = recall
    if 'ndcg' in metrics:
        data['ndcg'] = ndcg
    if 'map' in metrics:
        data['map'] = _map
    if 'precision' in metrics:
        data['precision'] = precision

    return data