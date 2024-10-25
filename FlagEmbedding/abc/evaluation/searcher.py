"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/searcher.py
"""
import os
import numpy as np
from typing import Any, Dict
from abc import ABC, abstractmethod

from FlagEmbedding.abc.inference import AbsEmbedder, AbsReranker
from FlagEmbedding.abc.evaluation.utils import index, search


class AbsEmbedder(ABC):
    """
    Base class for retrievers.
    Extend this class and implement __str__ and __call__ for custom retrievers.
    """
    def __init__(self, retriever: AbsEmbedder, search_top_k: int = 1000):
        self.retriever = retriever
        self.search_top_k = search_top_k

    # @abstractmethod
    def __str__(self) -> str:
        """
        Returns: str: Name of the retriever.
        """
        return self.retriever.model.config._name_or_path
        # pass

    # @abstractmethod
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        corpus_embd_save_dir: str = None,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        This is called during the retrieval process.
        
        Parameters:
            corpus: Dict[str, Dict[str, Any]]: Corpus of documents. 
                Structure: {<docid>: {"text": <text>}}.
                Example: {"doc-0": {"text": "This is a document."}}
            queries: Dict[str, str]: Queries to search for.
                Structure: {<qid>: <query>}.
                Example: {"q-0": "This is a query."}
            **kwargs: Any: Additional arguments.
        
        Returns: Dict[str, Dict[str, float]]: Top-k search results for each query. k is specified by search_top_k.
            Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
            Example: {"q-0": {"doc-0": 0.9}}
        """
        corpus_texts = [doc["text"] for _, doc in corpus.items()]
        queries_texts = [query for _, query in queries.items()]
        corpus_ids = list(corpus.keys())
        queries_ids = list(queries.keys())

        if corpus_embd_save_dir is not None:
            if os.path.exists(os.path.join(corpus_embd_save_dir, 'doc.npy')):
                corpus_emb = np.load(os.path.join(corpus_embd_save_dir, 'doc.npy'))
            else:
                corpus_emb = self.retriever.encode_corpus(corpus_texts, max_length=passage_max_length, **kwargs)
                if corpus_embd_save_dir is not None:
                    os.makedirs(corpus_embd_save_dir, exist_ok=True)
                    np.save(os.path.join(corpus_embd_save_dir, 'doc.npy'), corpus_emb)
        else:
            corpus_emb = self.retriever.encode_corpus(corpus_texts, max_length=passage_max_length, **kwargs)

        queries_emb = self.retriever.encode_queries(queries_texts, max_length=query_max_length, **kwargs)
        
        faiss_index = index(corpus_embeddings=corpus_emb)
        all_scores, all_indices = search(query_embeddings=queries_emb, faiss_index=faiss_index, k=self.search_top_k)

        results = {}
        for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            results[queries_ids[idx]] = {}
            for score, indice in zip(scores, indices):
                if corpus_ids[indice] != queries_ids[idx]:
                    results[queries_ids[idx]][corpus_ids[indice]] = float(score)

        return results

class AbsReranker(ABC):
    """
    Base class for rerankers.
    Extend this class and implement __str__ and __call__ for custom rerankers.
    """
    def __init__(self, reranker: AbsReranker, rerank_top_k: int = 100):
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    # @abstractmethod
    def __str__(self) -> str:
        """
        Returns: str: Name of the reranker.
        """
        return self.reranker.model.config._name_or_path
        # pass

    # @abstractmethod
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        search_results: Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        This is called during the reranking process.
        
        Parameters:
            corpus: Dict[str, Dict[str, Any]]: Corpus of documents. 
                Structure: {<docid>: {"text": <text>}}.
                Example: {"doc-0": {"text": "This is a document."}}
            queries: Dict[str, str]: Queries to search for.
                Structure: {<qid>: <query>}.
                Example: {"q-0": "This is a query."}
            search_results: Dict[str, Dict[str, float]]: Search results for each query.
                Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
                Example: {"q-0": {"doc-0": 0.9}}
            **kwargs: Any: Additional arguments.
        
        Returns: Dict[str, Dict[str, float]]: Reranked search results for each query. k is specified by rerank_top_k.
            Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
            Example: {"q-0": {"doc-0": 0.9}}
        """
        corpus_texts = [doc["text"] for _, doc in corpus.items()]
        queries_texts = [query for _, query in queries.items()]
        corpus_ids = list(corpus.keys())
        queries_ids = list(queries.keys())

        sentence_pairs = []

        for qid in search_results.keys():
            dids = list(search_results[qid].keys())[:self.rerank_top_k]
            for did in dids:
                sentence_pairs.append((queries[qid], corpus[did]["text"]))
        
        scores = self.reranker.compute_score(sentence_pairs, **kwargs)
        if isinstance(scores[0], list):
            scores = scores[0]
        
        # scores = np.asarray(scores)
        # scores = scores.reshape(-1, self.rerank_top_k)

        results = {}
        idx = 0
        for qid in search_results.keys():
            dids = list(search_results[qid].keys())[:self.rerank_top_k]
            results[qid] = {}
            for did in dids:
                results[qid][did] = float(scores[idx])
                idx += 1

        return results


