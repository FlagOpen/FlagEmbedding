from enum import Enum
from typing import Type
from dataclasses import dataclass
from collections import OrderedDict

from FlagEmbedding.abc.inference import AbsEmbedder
from FlagEmbedding.inference.embedder import FlagModel, BGEM3Model, FlagLLMModel, FlagICLModel


class PoolingMethod(Enum):
    LAST_TOKEN = "last_token"
    CLS = "cls"
    MEAN = "mean"


@dataclass
class EmbedderConfig:
    model_class: Type[AbsEmbedder]
    pooling_method: PoolingMethod
    trust_remote_code: bool = False


MODEL_MAPPING = OrderedDict([
    # ============================== BGE ==============================
    (
        "bge-en-icl", 
        EmbedderConfig(FlagICLModel, PoolingMethod.LAST_TOKEN)
    ),
    (
        "bge-multilingual-gemma2",
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN)
    ),
    (
        "bge-m3",
        EmbedderConfig(BGEM3Model, PoolingMethod.CLS)
    ),
    (
        "bge-large-en-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-base-en-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-small-en-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-large-zh-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-base-zh-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-small-zh-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-large-en",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-base-en",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-small-en",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-large-zh",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-base-zh",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        "bge-small-zh",
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    # ============================== E5 ==============================
    (
        "e5-mistral-7b-instruct",
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN)
    ),
    (
        "e5-large-v2",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "e5-base-v2",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "e5-small-v2",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "multilingual-e5-large-instruct",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "multilingual-e5-large",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "multilingual-e5-base",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "multilingual-e5-small",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "e5-large",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "e5-base",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        "e5-small",
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    # ============================== GTE ==============================
    (
        "gte-Qwen2-7B-instruct",
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN, True)
    ),
    (
        "gte-Qwen2-1.5B-instruct",
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN, True)
    ),
    (
        "gte-Qwen1.5-7B-instruct",
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN, True)
    ),
    (
        "gte-multilingual-base",
        EmbedderConfig(FlagModel, PoolingMethod.CLS, True)
    ),
    (
        "gte-large-en-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS, True)
    ),
    (
        "gte-base-en-v1.5",
        EmbedderConfig(FlagModel, PoolingMethod.CLS, True)
    ),
    (
        'gte-large',
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        'gte-base',
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        'gte-small',
        EmbedderConfig(FlagModel, PoolingMethod.MEAN)
    ),
    (
        'gte-large-zh',
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        'gte-base-zh',
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    (
        'gte-small-zh',
        EmbedderConfig(FlagModel, PoolingMethod.CLS)
    ),
    # ============================== SFR ==============================
    (
        'SFR-Embedding-2_R',
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN)
    ),
    (
        'SFR-Embedding-Mistral',
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN)
    ),
    # ============================== Linq ==============================
    (
        'Linq-Embed-Mistral',
        EmbedderConfig(FlagLLMModel, PoolingMethod.LAST_TOKEN)
    )
    # TODO: Add more models, such as Jina, Stella_v5, NV-Embed, etc.
])
