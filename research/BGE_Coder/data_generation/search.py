from typing import Optional, List

import faiss
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagModel

def create_index(embeddings: np.ndarray, use_gpu: bool = False):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def search(
        faiss_index: faiss.Index,
        k: int = 100,
        query_embeddings: Optional[np.ndarray] = None,
        load_path: Optional[str] = None
):
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

def get_top1(
        small_docs,
        encoder_name,
        docs: List[str],
        top: int = 1
):
    encoder = FlagModel(encoder_name, trust_remote_code=True)
    doc_emb = encoder.encode_corpus(docs, max_length=512, batch_size=256)
    small_doc_emb = encoder.encode_corpus(small_docs, max_length=512, batch_size=256)
    faiss_index = create_index(doc_emb, True)
    all_scores, all_indices = search(faiss_index, 1000, small_doc_emb)
    return_docs = []
    for i in range(len(all_indices)):
        return_docs.append([])
        for idx, score in zip(all_indices[i][20:], all_scores[i][20:]):
            d1 = set(docs[idx].split())
            d2 = set(small_docs[i].split())
            if len(d1 & d2) / len(d1 | d2) > 0.95:
                continue
            return_docs[-1].append(docs[idx])
            if len(return_docs[-1]) >= top:
                break
        if len(return_docs[-1]) == 0:
            print(all_indices[i], all_scores[i])
    # print(return_docs)
    del faiss_index
    return return_docs
