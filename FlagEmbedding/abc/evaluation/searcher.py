"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/searcher.py
"""
import os
import logging
import gc
import torch
import numpy as np
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from FlagEmbedding.abc.inference import AbsEmbedder, AbsReranker
from FlagEmbedding.abc.evaluation.utils import index, search

logger = logging.getLogger(__name__)


class EvalRetriever(ABC):
    """
    This is the base class for retriever.
    """
    def __init__(self, embedder: AbsEmbedder, search_top_k: int = 1000, overwrite: bool = False):
        self.embedder = embedder
        self.search_top_k = search_top_k
        self.overwrite = overwrite

    def __str__(self) -> str:
        """
        Returns: str: Name of the retriever.
        """
        return os.path.basename(self.embedder.model.config._name_or_path)

    def stop_multi_process_pool(self):
        self.embedder.stop_self_pool()
        # if self.embedder.pool is not None:
        #     self.embedder.stop_multi_process_pool(self.embedder.pool)
        #     self.embedder.pool = None
        #     self.embedder.model.to('cpu')
        #     gc.collect()
        #     torch.cuda.empty_cache()

    @abstractmethod
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Abstract method to be overrode. This is called during the retrieval process.
        
        Parameters:
            corpus: Dict[str, Dict[str, Any]]: Corpus of documents. 
                Structure: {<docid>: {"text": <text>}}.
                Example: {"doc-0": {"text": "This is a document."}}
            queries: Dict[str, str]: Queries to search for.
                Structure: {<qid>: <query>}.
                Example: {"q-0": "This is a query."}
            corpus_embd_save_dir (Optional[str]): Defaults to :data:`None`.
            ignore_identical_ids (bool): Defaults to :data:`False`.
            **kwargs: Any: Additional arguments.
        
        Returns: Dict[str, Dict[str, float]]: Top-k search results for each query. k is specified by search_top_k.
            Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
            Example: {"q-0": {"doc-0": 0.9}}
        """


class EvalDenseRetriever(EvalRetriever):
    """
    Child class of :class:EvalRetriever for dense retrieval.
    """
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
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
            corpus_embd_save_dir (Optional[str]): Defaults to :data:`None`.
            ignore_identical_ids (bool): Defaults to :data:`False`.
            **kwargs: Any: Additional arguments.
        
        Returns: Dict[str, Dict[str, float]]: Top-k search results for each query. k is specified by search_top_k.
            Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
            Example: {"q-0": {"doc-0": 0.9}}
        """
        if ignore_identical_ids:
            logger.warning("ignore_identical_ids is set to True. This means that the search results will not contain identical ids. Note: Dataset such as MIRACL should NOT set this to True.")

        # dense embedding models do not require language as input: AIRBench evaluation
        kwargs.pop("language", None)

        corpus_ids = []
        corpus_texts = []
        for docid, doc in corpus.items():
            corpus_ids.append(docid)
            corpus_texts.append(
                doc["text"] if "title" not in doc 
                else f"{doc['title']} {doc['text']}".strip()
            )
        queries_ids = []
        queries_texts = []
        for qid, query in queries.items():
            queries_ids.append(qid)
            queries_texts.append(query)

        if corpus_embd_save_dir is not None:
            if os.path.exists(os.path.join(corpus_embd_save_dir, "doc.npy")) and not self.overwrite:
                corpus_emb = np.load(os.path.join(corpus_embd_save_dir, "doc.npy"))
            else:
                corpus_emb = self.embedder.encode_corpus(corpus_texts, **kwargs)
        else:
            corpus_emb = self.embedder.encode_corpus(corpus_texts, **kwargs)

        queries_emb = self.embedder.encode_queries(queries_texts, **kwargs)

        # check if the embeddings are in dictionary format: M3Embedder
        if isinstance(corpus_emb, dict):
            corpus_emb = corpus_emb["dense_vecs"]
        if isinstance(queries_emb, dict):
            queries_emb = queries_emb["dense_vecs"]
        
        if corpus_embd_save_dir is not None and \
            (not os.path.exists(os.path.join(corpus_embd_save_dir, "doc.npy")) or self.overwrite):
            os.makedirs(corpus_embd_save_dir, exist_ok=True)
            np.save(os.path.join(corpus_embd_save_dir, "doc.npy"), corpus_emb)
        
        gc.collect()
        torch.cuda.empty_cache()

        faiss_index = index(corpus_embeddings=corpus_emb)
        all_scores, all_indices = search(query_embeddings=queries_emb, faiss_index=faiss_index, k=self.search_top_k)

        results = {}
        for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            results[queries_ids[idx]] = {}
            for score, indice in zip(scores, indices):
                if indice != -1:
                    if ignore_identical_ids and corpus_ids[indice] == queries_ids[idx]:
                        continue
                    results[queries_ids[idx]][corpus_ids[indice]] = float(score)

        return results


class EvalReranker:
    """
    Class for reranker during evaluation.
    """
    def __init__(self, reranker: AbsReranker, rerank_top_k: int = 100):
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def __str__(self) -> str:
        """
        Returns: str: Name of the reranker.
        """
        return os.path.basename(self.reranker.model.config._name_or_path)

    def stop_multi_process_pool(self):
        self.reranker.stop_self_pool()
        # if self.reranker.pool is not None:
        #     self.reranker.stop_multi_process_pool(self.reranker.pool)
        #     self.reranker.pool = None
        #     self.reranker.model.to('cpu')
        #     gc.collect()
        #     torch.cuda.empty_cache()

    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        search_results: Dict[str, Dict[str, float]],
        ignore_identical_ids: bool = False,
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
        # truncate search results to top_k
        for qid in search_results:
            search_results[qid] = dict(
                sorted(search_results[qid].items(), key=lambda x: x[1], reverse=True)[
                    :self.rerank_top_k
                ]
            )
        # generate sentence pairs
        sentence_pairs = []
        pairs = []
        for qid in search_results:
            for docid in search_results[qid]:
                if ignore_identical_ids and qid == docid:
                    continue
                sentence_pairs.append(
                    {
                        "qid": qid,
                        "docid": docid,
                        "query": queries[qid],
                        "doc": corpus[docid]["text"] if "title" not in corpus[docid] 
                            else f"{corpus[docid]['title']} {corpus[docid]['text']}".strip(),
                    }
                )
                pairs.append(
                    (
                        queries[qid],
                        corpus[docid]["text"] if "title" not in corpus[docid] 
                            else f"{corpus[docid]['title']} {corpus[docid]['text']}".strip()
                    )
                )
        # compute scores
        scores = self.reranker.compute_score(pairs)
        for i, score in enumerate(scores):
            sentence_pairs[i]["score"] = float(score)
        # rerank
        reranked_results = {qid: {} for qid in search_results}
        for pair in sentence_pairs:
            reranked_results[pair["qid"]][pair["docid"]] = pair["score"]
        return reranked_results
