from typing import Type
from dataclasses import dataclass
from collections import OrderedDict

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.inference.reranker import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker, LightWeightFlagLLMReranker


@dataclass
class RerankerConfig:
    model_class: Type[AbsReranker]
    trust_remote_code: bool = False


MODEL_MAPPING = OrderedDict([
    # ============================== BGE ==============================
    (
        "bge-reranker-base", 
        RerankerConfig(FlagReranker)
    ),
    (
        "bge-reranker-large", 
        RerankerConfig(FlagReranker)
    ),
    (
        "bge-reranker-v2-m3",
        RerankerConfig(FlagReranker)
    ),
    (
        "bge-reranker-v2-gemma",
        RerankerConfig(FlagLLMReranker)
    ),
    (
        "bge-reranker-v2-minicpm-layerwise",
        RerankerConfig(LayerWiseFlagLLMReranker)
    ),
    (
        "bge-reranker-v2.5-gemma2-lightweight",
        RerankerConfig(LightWeightFlagLLMReranker)
    ),
    # TODO: Add more models, such as Jina, Stella_v5, NV-Embed, etc.
])
