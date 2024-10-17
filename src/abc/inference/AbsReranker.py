from typing import Any, Union, List, Tuple
from abc import ABC, abstractmethod


class AbsReranker(ABC):
    """
    Base class for embedder.
    Extend this class and implement `encode_queries`, `encode_passages`, `encode` for custom embedders.
    """
    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
    
    @abstractmethod
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = False,
        **kwargs: Any,
    ):
        """
        This method should encode sentences and return embeddings.
        """
        pass
