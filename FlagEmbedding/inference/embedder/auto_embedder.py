import os

from .model_mapping import MODEL_MAPPING


class FlagAutoModel:
    def __init__(self):
        raise EnvironmentError(
            "FlagAutoModel is designed to be instantiated using the `FlagAutoModel.from_finetuned(model_name_or_path)` method."
        )
    
    @classmethod
    def from_finetuned(
        cls,
        model_name_or_path: str,
        normalize_embeddings: bool = False,
        use_fp16: bool = False,
        **kwargs,
    ):
        model_name = os.path.basename(model_name_or_path)
        if model_name.startswith("checkpoint-"):
            model_name = os.path.basename(os.path.dirname(model_name_or_path))
        
        if model_name not in MODEL_MAPPING:
            raise ValueError(
                f"Model name '{model_name}' not found in the model mapping. You can pull request to add the model to "
                "`https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/model_mapping.py`. " 
                "If need, you can create a new `<model>.py` file in `https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/encoder_only` "
                "or `https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/decoder_only`. "
                "Welcome to contribute! You can also directly use the corresponding model class to instantiate the model."
            )
        
        model_config = MODEL_MAPPING[model_name]
        
        model_class = model_config.model_class
        pooling_method = kwargs.pop("pooling_method", model_config.pooling_method.value)
        trust_remote_code = kwargs.pop("trust_remote_code", model_config.trust_remote_code)
        
        return model_class(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            pooling_method=pooling_method,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
