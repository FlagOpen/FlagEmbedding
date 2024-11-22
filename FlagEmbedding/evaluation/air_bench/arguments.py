from dataclasses import dataclass, field
from typing import List, Optional
from air_benchmark import EvalArgs as AIRBenchEvalArgs


@dataclass
class AIRBenchEvalModelArgs:
    """
    Evaluation Model arguments for AIR Bench.
    """
    embedder_name_or_path: str = field(
        metadata={"help": "The embedder name or path.", "required": True}
    )
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl"]}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    pooling_method: str = field(
        default="cls", metadata={"help": "The pooling method fot the embedder."}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    examples_for_task: Optional[str] = field(
        default=None, metadata={"help": "Examples for task"}
    )
    examples_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format for examples instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    reranker_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The reranker name or path."}
    )
    reranker_model_class: Optional[str] = field(
        default=None, metadata={"help": "The reranker model class. Available classes: ['encoder-only-base', 'decoder-only-base', 'decoder-only-layerwise', 'decoder-only-lightweight']. Default: None. For the custom model, you need to specify the model class.", "choices": ["encoder-only-base", "decoder-only-base", "decoder-only-layerwise", "decoder-only-lightweight"]}
    )
    reranker_peft_path: Optional[str] = field(
        default=None, metadata={"help": "The reranker peft path."}
    )
    use_bf16: bool = field(
        default=False, metadata={"help": "whether to use bf16 for inference"}
    )
    query_instruction_for_rerank: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    passage_instruction_for_rerank: Optional[str] = field(
        default=None, metadata={"help": "Instruction for passage"}
    )
    passage_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for passage instruction"}
    )
    model_cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    embedder_batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    reranker_batch_size: int = field(
        default=3000, metadata={"help": "Batch size for inference."}
    )
    embedder_query_max_length: int = field(
        default=512, metadata={"help": "Max length for query."}
    )
    embedder_passage_max_length: int = field(
        default=512, metadata={"help": "Max length for passage."}
    )
    reranker_query_max_length: Optional[int] = field(
        default=None, metadata={"help": "Max length for reranking."}
    )
    reranker_max_length: int = field(
        default=512, metadata={"help": "Max length for reranking."}
    )
    normalize: bool = field(
        default=False, metadata={"help": "whether to normalize the reranking scores"}
    )
    prompt: Optional[str] = field(
        default=None, metadata={"help": "The prompt for the reranker."}
    )
    cutoff_layers: List[int] = field(
        default=None, metadata={"help": "The output layers of layerwise/lightweight reranker."}
    )
    compress_ratio: int = field(
        default=1, metadata={"help": "The compress ratio of lightweight reranker."}
    )
    compress_layers: Optional[int] = field(
        default=None, metadata={"help": "The compress layers of lightweight reranker.", "nargs": "+"}
    )
