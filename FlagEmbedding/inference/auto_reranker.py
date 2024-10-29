import os
import logging
from typing import Union, Optional

from FlagEmbedding.inference.reranker.model_mapping import (
    RerankerModelClass,
    RERANKER_CLASS_MAPPING,
    AUTO_RERANKER_MAPPING
)

logger = logging.getLogger(__name__)


class FlagAutoReranker:
    def __init__(self):
        raise EnvironmentError(
            "FlagAutoReranker is designed to be instantiated using the `FlagAutoReranker.from_finetuned(model_name_or_path)` method."
        )

    @classmethod
    def from_finetuned(
        cls,
        model_name_or_path: str,
        model_class: Optional[Union[str, RerankerModelClass]] = None,
        use_fp16: bool = False,
        trust_remote_code: Optional[bool] = None,
        **kwargs,
    ):
        model_name = os.path.basename(model_name_or_path)
        if model_name.startswith("checkpoint-"):
            model_name = os.path.basename(os.path.dirname(model_name_or_path))

        if model_class is not None and model_class != 'auto':
            _model_class = RERANKER_CLASS_MAPPING[RerankerModelClass(model_class)]
            if trust_remote_code is None:
                trust_remote_code = False
            logging.warning(
                f"`trust_remote_code` is not specified, set to default value '{trust_remote_code}'."
            )
        else:
            if model_name not in AUTO_RERANKER_MAPPING:
                raise ValueError(
                    f"Model name '{model_name}' not found in the model mapping. You can pull request to add the model to "
                    "`https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/reranker/model_mapping.py`. " 
                    "If need, you can create a new `<model>.py` file in `https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/reranker/encoder_only` "
                    "or `https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/reranker/decoder_only`. "
                    "Welcome to contribute! You can also directly specify the corresponding `model_class` to instantiate the model."
                )

            model_config = AUTO_RERANKER_MAPPING[model_name]

            _model_class = model_config.model_class
            if trust_remote_code is None:
                trust_remote_code = model_config.trust_remote_code

        return _model_class(
            model_name_or_path,
            use_fp16=use_fp16,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )