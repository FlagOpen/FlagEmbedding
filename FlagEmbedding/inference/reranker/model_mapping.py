from enum import Enum
from typing import Type
from dataclasses import dataclass
from collections import OrderedDict

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.inference.reranker import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker, LightWeightFlagLLMReranker


class RerankerModelClass(Enum):
    ENCODER_ONLY_BASE = "encoder-only-base"
    DECODER_ONLY_BASE = "decoder-only-base"
    DECODER_ONLY_LAYERWISE = "decoder-only-layerwise"
    DECODER_ONLY_LIGHTWEIGHT = "decoder-only-lightweight"


RERANKER_CLASS_MAPPING = OrderedDict([
    (RerankerModelClass.ENCODER_ONLY_BASE, FlagReranker),
    (RerankerModelClass.DECODER_ONLY_BASE, FlagLLMReranker),
    (RerankerModelClass.DECODER_ONLY_LAYERWISE, LayerWiseFlagLLMReranker),
    (RerankerModelClass.DECODER_ONLY_LIGHTWEIGHT, LightWeightFlagLLMReranker)
])


@dataclass
class RerankerConfig:
    model_class: Type[AbsReranker]
    trust_remote_code: bool = False


AUTO_RERANKER_MAPPING = OrderedDict([
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
    # others
    (
        "jinaai/jina-reranker-v2-base-multilingual",
        RerankerConfig(FlagReranker)
    ),
    (
        "Alibaba-NLP/gte-multilingual-reranker-base",
        RerankerConfig(FlagReranker)
    ),
    (
        "maidalun1020/bce-reranker-base_v1",
        RerankerConfig(FlagReranker)
    ),
    (
        "jinaai/jina-reranker-v1-turbo-en",
        RerankerConfig(FlagReranker)
    ),
    # TODO: Add more models.
])
