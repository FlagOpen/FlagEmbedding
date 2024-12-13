import os
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel


def pad_colbert_vecs(colbert_vecs_list, device):
    lengths = [vec.shape[0] for vec in colbert_vecs_list]
    max_len = max(lengths)
    dim = colbert_vecs_list[0].shape[1]

    padded_tensor = torch.zeros(len(colbert_vecs_list), max_len, dim, dtype=torch.float, device=device)
    for i, vec in enumerate(colbert_vecs_list):
        length = vec.shape[0]
        padded_tensor[i, :length, :] = torch.tensor(vec, dtype=torch.float, device=device)

    return padded_tensor


def compute_colbert_scores(query_colbert_vecs, passage_colbert_vecs):
    # query_colbert_vecs: (Q, Tq, D)
    # passage_colbert_vecs: (P, Tp, D)
    # einsum 식에서 q:queries, p:passages, r:query tokens dim, c:passage tokens dim, d:embedding dim
    dot_products = torch.einsum("qrd,pcd->qprc", query_colbert_vecs, passage_colbert_vecs)  # Q,P,Tq,Tp
    max_per_query_token, _ = dot_products.max(dim=3)  # max over c (Tp)
    colbert_scores = max_per_query_token.sum(dim=2)  # sum over r (Tq)
    return colbert_scores


def hybrid_dbfs_ensemble(dense_scores, sparse_scores, colbert_scores, weights=(0.33, 0.33, 0.34)):
    w_dense, w_sparse, w_colbert = weights
    # 모든 입력이 torch.Tensor일 경우 아래 연산 정상 작동
    return w_dense * dense_scores + w_sparse * sparse_scores + w_colbert * colbert_scores


def test_m3_single_device():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        devices="cuda:0",
        pooling_method='cls',
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )

    queries = [
                  "What is BGE M3?",
                  "Defination of BM25"
              ] * 100
    passages = [
                   "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
                   "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"
               ] * 100

    queries_embeddings = model.encode_queries(
        queries,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    passages_embeddings = model.encode_corpus(
        passages,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    # device 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dense_vecs, lexical_weights 등이 numpy array 형태일 수 있으므로 텐서로 변환
    q_dense = torch.tensor(queries_embeddings["dense_vecs"], dtype=torch.float, device=device)
    p_dense = torch.tensor(passages_embeddings["dense_vecs"], dtype=torch.float, device=device)
    dense_scores = q_dense @ p_dense.T

    # sparse_scores도 numpy array를 텐서로 변환
    sparse_scores_np = model.compute_lexical_matching_score(
        queries_embeddings["lexical_weights"],
        passages_embeddings["lexical_weights"]
    )
    sparse_scores = torch.tensor(sparse_scores_np, dtype=torch.float, device=device)

    # colbert_vecs 패딩 후 텐서 변환
    query_colbert_vecs = pad_colbert_vecs(queries_embeddings["colbert_vecs"], device)
    passage_colbert_vecs = pad_colbert_vecs(passages_embeddings["colbert_vecs"], device)

    colbert_scores = compute_colbert_scores(query_colbert_vecs, passage_colbert_vecs)

    # 모든 스코어가 torch.Tensor이므로 오류 없이 연산 가능
    hybrid_scores = hybrid_dbfs_ensemble(dense_scores, sparse_scores, colbert_scores)

    print("Dense score:\n", dense_scores[:2, :2])
    print("Sparse score:\n", sparse_scores[:2, :2])
    print("ColBERT score:\n", colbert_scores[:2, :2])
    print("Hybrid DBSF Ensemble score:\n", hybrid_scores[:2, :2])


if __name__ == '__main__':
    test_m3_single_device()
    print("--------------------------------")
    print("Expected Output for Dense & Sparse (original):")
    print("Dense score:")
    print(" [[0.626  0.3477]\n [0.3496 0.678 ]]")
    print("Sparse score:")
    print(" [[0.19554901 0.00880432]\n [0.         0.18036556]]")
    print("--------------------------------")
    print("ColBERT and Hybrid DBSF scores will vary depending on the actual embeddings.")
