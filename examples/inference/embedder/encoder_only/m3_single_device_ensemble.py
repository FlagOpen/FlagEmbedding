import os
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel


def pad_colbert_vecs(colbert_vecs_list, device):
    """
    Since ColBERT embeddings are computed on a token-level basis, each document (or query)
    may produce a different number of token embeddings. This function aligns all embeddings
    to the same length by padding shorter sequences with zeros, ensuring that every input
    ends up with a uniform shape.

    Steps:
    1. Determine the maximum sequence length (i.e., the largest number of tokens in any
       query or passage within the batch).
    2. For each set of token embeddings, pad it with zeros until it matches the max
       sequence length. Zeros here act as placeholders and do not affect the similarity
       computations since they represent "no token."
    3. Convert all padded embeddings into a single, consistent tensor and move it to the
       specified device (e.g., GPU) for efficient batch computation.

    By performing this padding operation, subsequent tensor operations (like the einsum
    computations for ColBERT scoring) become simpler and more efficient, as all sequences
    share a common shape.
    """

    lengths = [vec.shape[0] for vec in colbert_vecs_list]
    max_len = max(lengths)
    dim = colbert_vecs_list[0].shape[1]

    padded_tensor = torch.zeros(len(colbert_vecs_list), max_len, dim, dtype=torch.float, device=device)
    for i, vec in enumerate(colbert_vecs_list):
        length = vec.shape[0]
        padded_tensor[i, :length, :] = torch.tensor(vec, dtype=torch.float, device=device)

    return padded_tensor


def compute_colbert_scores(query_colbert_vecs, passage_colbert_vecs):
    """
    Compute ColBERT scores:

    ColBERT (Contextualized Late Interaction over BERT) evaluates the similarity
    between a query and a passage at the token level. Instead of producing a single
    dense vector for each query or passage, ColBERT maintains embeddings for every
    token. This allows for finer-grained matching, capturing more subtle similarities.

    Definitions of variables:
    - q: Number of queries (Q)
    - p: Number of passages (P)
    - r: Number of tokens in each query (Tq)
    - c: Number of tokens in each passage (Tp)
    - d: Embedding dimension (D)

    I used the operation `einsum("qrd,pcd->qprc", query_colbert_vecs, passage_colbert_vecs)`:
    - einsum (Einstein summation) is a powerful notation and function for
      expressing and computing multi-dimensional tensor contractions. It allows you
      to specify how dimensions in input tensors correspond to each other and how
      they should be combined (multiplied and summed) to produce the output.

    In this particular case:
    - "qrd" corresponds to (Q, Tq, D) for query token embeddings.
    - "pcd" corresponds to (P, Tp, D) for passage token embeddings.
    - "qrd,pcd->qprc" means:
      1. For each query q and passage p, compute the dot product between every query token
         embedding (r) and every passage token embedding (c) across the embedding dimension d.
      2. This results in a (Q, P, Tq, Tp) tensor (qprc), where each element is the similarity
         score between a single query token and a single passage token.

    After computing this full matrix of token-to-token scores:
    - We take the maximum over the passage token dimension (c) for each query token (r).
      This step identifies, for each query token, which passage token is the "best match."
    - Then we sum over all query tokens (r) to aggregate their best matches into a single
      score per query-passage pair.

    In summary:
    1. einsum to get all pairwise token similarities.
    2. max over passage tokens to find the best matching passage token for each query token.
    3. sum over query tokens to combine all the best matches into a final ColBERT score
       for each query-passage pair.
    """

    dot_products = torch.einsum("qrd,pcd->qprc", query_colbert_vecs, passage_colbert_vecs)  # Q,P,Tq,Tp
    max_per_query_token, _ = dot_products.max(dim=3)
    colbert_scores = max_per_query_token.sum(dim=2)
    return colbert_scores


def hybrid_dbfs_ensemble_simple_linear_combination(dense_scores, sparse_scores, colbert_scores, weights=(0.45, 0.45, 0.1)):
    w_dense, w_sparse, w_colbert = weights
    return w_dense * dense_scores + w_sparse * sparse_scores + w_colbert * colbert_scores


def test_m3_single_device():
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        devices="cuda:0",
        pooling_method='cls',
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )

    queries = [
                  "What is Sionic AI?",
                  "Try https://sionicstorm.ai today!"
              ] * 100
    passages = [
                   "Sionic AI delivers more accessible and cost-effective AI technology addressing the various needs to boost productivity and drive innovation.",
                   "The Large Language Model (LLM) is not for research and experimentation. We offer solutions that leverage LLM to add value to your business. Anyone can easily train and control AI."
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    q_dense = torch.tensor(queries_embeddings["dense_vecs"], dtype=torch.float, device=device)
    p_dense = torch.tensor(passages_embeddings["dense_vecs"], dtype=torch.float, device=device)
    dense_scores = q_dense @ p_dense.T

    sparse_scores_np = model.compute_lexical_matching_score(
        queries_embeddings["lexical_weights"],
        passages_embeddings["lexical_weights"]
    )

    sparse_scores = torch.tensor(sparse_scores_np, dtype=torch.float, device=device)

    query_colbert_vecs = pad_colbert_vecs(queries_embeddings["colbert_vecs"], device)
    passage_colbert_vecs = pad_colbert_vecs(passages_embeddings["colbert_vecs"], device)
    colbert_scores = compute_colbert_scores(query_colbert_vecs, passage_colbert_vecs)

    hybrid_scores = hybrid_dbfs_ensemble_simple_linear_combination(dense_scores, sparse_scores, colbert_scores)

    print("Dense score:\n", dense_scores[:2, :2])
    print("Sparse score:\n", sparse_scores[:2, :2])
    print("ColBERT score:\n", colbert_scores[:2, :2])
    print("Hybrid DBSF Ensemble score:\n", hybrid_scores[:2, :2])


if __name__ == '__main__':
    test_m3_single_device()
    print("Expected Vector Scores")
    print("--------------------------------")
    print("Dense score:")
    print(" [[0.626  0.3477]\n [0.3496 0.678 ]]")
    print("Sparse score:")
    print(" [[0.19554901 0.00880432]\n [0.         0.18036556]]")
    print("ColBERT score:")
    print("[[5.8061, 3.1195] \n [5.6822, 4.6513]]")
    print("Hybrid DBSF Ensemble score:")
    print("[[0.9822, 0.5125] \n [0.8127, 0.6958]]")
    print("--------------------------------")
