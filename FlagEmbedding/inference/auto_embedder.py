import os
import logging
from typing import List, Union, Optional

from FlagEmbedding.inference.embedder.model_mapping import (
    EmbedderModelClass,
    AUTO_EMBEDDER_MAPPING, EMBEDDER_CLASS_MAPPING
)

logger = logging.getLogger(__name__)


class FlagAutoModel:
    def __init__(self):
        raise EnvironmentError(
            "FlagAutoModel is designed to be instantiated using the `FlagAutoModel.from_finetuned(model_name_or_path)` method."
        )

    @classmethod
    def from_finetuned(
        cls,
        model_name_or_path: str,
        model_class: Optional[Union[str, EmbedderModelClass]] = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        devices: Optional[Union[str, List[str]]] = None,
        pooling_method: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        query_instruction_format: Optional[str] = None,
        **kwargs,
    ):
        model_name = os.path.basename(model_name_or_path)
        if model_name.startswith("checkpoint-"):
            model_name = os.path.basename(os.path.dirname(model_name_or_path))

        if model_class is not None and model_class != 'auto':
            _model_class = EMBEDDER_CLASS_MAPPING[EmbedderModelClass(model_class)]
            if pooling_method is None:
                pooling_method = _model_class.DEFAULT_POOLING_METHOD
                logger.warning(
                    f"`pooling_method` is not specified, use default pooling method '{pooling_method}'."
                )
            if trust_remote_code is None:
                trust_remote_code = False
                logger.warning(
                    f"`trust_remote_code` is not specified, set to default value '{trust_remote_code}'."
                )
            if query_instruction_format is None:
                query_instruction_format = "{}{}"
                logger.warning(
                    f"`query_instruction_format` is not specified, set to default value '{query_instruction_format}'."
                )
        else:
            if model_name not in AUTO_EMBEDDER_MAPPING:
                raise ValueError(
                    f"Model name '{model_name}' not found in the model mapping. You can pull request to add the model to "
                    "`https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/model_mapping.py`. " 
                    "If need, you can create a new `<model>.py` file in `https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/encoder_only` "
                    "or `https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/decoder_only`. "
                    "Welcome to contribute! You can also directly specify the corresponding `model_class` to instantiate the model."
                )

            model_config = AUTO_EMBEDDER_MAPPING[model_name]

            _model_class = model_config.model_class
            if pooling_method is None:
                pooling_method = model_config.pooling_method.value
            if trust_remote_code is None:
                trust_remote_code = model_config.trust_remote_code
            if query_instruction_format is None:
                query_instruction_format = model_config.query_instruction_format

        return _model_class(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            pooling_method=pooling_method,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )