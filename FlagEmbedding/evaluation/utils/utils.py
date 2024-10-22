import os
import faiss
import numpy as np
from tqdm import tqdm


def index(
    index_factory: str = "Flat", 
    corpus_embeddings: np.array = None, 
    load_path: str = None
):
    if corpus_embeddings is None:
        corpus_embeddings = np.load(load_path)
    print(corpus_embeddings.shape)
    # create faiss index
    print('create faiss index...')
    faiss_index = faiss.index_factory(corpus_embeddings.shape[1], index_factory, faiss.METRIC_INNER_PRODUCT)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    print('add data...')
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    print('add over...')
    
    return faiss_index


def search(
    faiss_index: faiss.Index, 
    k: int = 100, 
    query_embeddings: np.array = None,
    load_path: str = None
):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    if query_embeddings is None:
        query_embeddings = np.load(load_path)

    query_size = len(query_embeddings)

    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, 32), desc="Searching"):
        j = min(i + 32, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices