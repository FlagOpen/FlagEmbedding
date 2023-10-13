import os
from typing import Optional, List
from dataclasses import dataclass, field
from sentence_transformers import models, SentenceTransformer
from transformers import HfArgumentParser


def convert_ours_ckpt_to_sentence_transformer(src_dir, dest_dir, pooling_method: List[str] = ['cls'], dense_metric: str="cos"):
    assert os.path.exists(src_dir), f"Make sure the encoder path {src_dir} is valid on disk!"
    assert "decoder" not in pooling_method, f"Pooling method 'decode' cannot be saved as sentence_transformers because it uses the decoder stack to produce sentence embedding."
    if dest_dir is None:
        dest_dir = src_dir

    print(f"loading model from {src_dir} and saving the sentence_transformer model at {dest_dir}...")

    word_embedding_model = models.Transformer(src_dir)
    modules = [word_embedding_model]
    ndim = word_embedding_model.get_word_embedding_dimension()

    if "cls" in pooling_method:
        pooling_model = models.Pooling(ndim, pooling_mode="cls")
        pooling_method.remove("cls")
    elif "mean" in pooling_method:
        pooling_model = models.Pooling(ndim, pooling_mode="mean")
        pooling_method.remove("mean")
    else:
        raise NotImplementedError(f"Fail to find cls or mean in pooling_method {pooling_method}!")
    
    modules.append(pooling_model)

    if "dense" in pooling_method:
        modules.append(models.Dense(ndim, ndim, bias=False))
        pooling_method.remove("dense")
    
    assert len(pooling_method) == 0, f"Found unused pooling_method {pooling_method}!"

    if dense_metric == "cos":
        normalize_layer = models.Normalize()
        modules.append(normalize_layer)

    model = SentenceTransformer(modules=modules, device='cpu')
    model.save(dest_dir)


@dataclass
class Args:
    encoder: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the encoder model.'}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the output sentence_transformer model.'}
    )
    pooling_method: List[str] = field(
        default_factory=lambda: ["cls"],
        metadata={'help': 'Pooling methods to aggregate token embeddings for a sequence embedding. {cls, mean, dense, decoder}'}
    )
    dense_metric: str = field(
        default="cos",
        metadata={'help': 'What type of metric for dense retrieval? ip, l2, or cos.'}
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Cache folder for huggingface transformers.'}
    )

    def __post_init__(self):
        convert_ours_ckpt_to_sentence_transformer(self.encoder, self.output_dir, self.pooling_method, self.dense_metric)

if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()

