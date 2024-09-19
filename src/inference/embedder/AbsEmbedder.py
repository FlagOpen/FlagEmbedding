from typing import Any, Union, List
from abc import ABC, abstractmethod


class AbsEmbedder(ABC):
    """
    Base class for embedder.
    Extend this class and implement `encode_queries`, `encode_passages`, `encode` for custom embedders.
    """
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = False,
        use_fp16: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
    
    @abstractmethod
    def encode_queries(
        self,
        queries: Union[List[str], str],
        **kwargs: Any,
    ):
        """
        This method should encode queries and return embeddings.
        """
        pass
    
    @abstractmethod
    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        **kwargs: Any,
    ):
        """
        This method should encode corpus and return embeddings.
        """
        pass
    
    @abstractmethod
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        **kwargs: Any,
    ):
        """
        This method should encode sentences and return embeddings.
        """
        pass
