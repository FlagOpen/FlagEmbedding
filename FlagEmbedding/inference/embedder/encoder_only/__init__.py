from .base import BaseEmbedder as FlagModel
from .m3 import M3Embedder as BGEM3FlagModel

__all__ = [
    "FlagModel",
    "BGEM3FlagModel",
]
