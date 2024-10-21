"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/searcher.py
"""
from typing import Any, Dict
from abc import ABC, abstractmethod


class AbsRetriever(ABC):
    """
    Base class for retrievers.
    Extend this class and implement __str__ and __call__ for custom retrievers.
    """
    def __init__(self, search_top_k: int = 1000):
        self.search_top_k = search_top_k

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns: str: Name of the retriever.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
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
        pass

class AbsReranker(ABC):
    """
    Base class for rerankers.
    Extend this class and implement __str__ and __call__ for custom rerankers.
    """
    def __init__(self, rerank_top_k: int = 100):
        self.rerank_top_k = rerank_top_k

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns: str: Name of the reranker.
        """
        pass

    @abstractmethod
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
        pass
