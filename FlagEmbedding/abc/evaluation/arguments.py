"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/evaluation_arguments.py
"""
from dataclasses import dataclass, field


@dataclass
class AbsEvalArgs:
    dataset_dir: str = field(
        metadata={"help": "Path to the dataset directory. The data directory should contain the following files: corpus.jsonl, <split1>_queries.jsonl, ..., <splitN>_queries.jsonl, <split1>_qrels.jsonl, ..., <splitN>_qrels.jsonl."}
    )
    splits: str = field(
        default='test', metadata={"help": "Splits to evaluate. Default: test", "nargs": "+"}
    )
    corpus_embd_save_dir: str = field(
        default=None, metadata={"help": "Path to save corpus embeddings. If None, embeddings are not saved."}
    )
    output_dir: str = field(
        default="./search_results", metadata={"help": "Path to save results."}
    )
    search_top_k: int = field(
        default=1000, metadata={"help": "Top k values for evaluation."}
    )
    rerank_top_k: int = field(default=100, metadata={"help": "Top k for reranking."})
    cache_path: str = field(
        default=None, metadata={"help": "Cache directory for datasets."}
    )
    overwrite: bool = field(
        default=False, metadata={"help": "whether to overwrite evaluation results"}
    )

@dataclass
class AbsModelArgs:
    embedder_name_or_path: str = field(
        metadata={"help": "The embedder name or path."}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: List[str] = field(
        default=None, metadata={"help": "Devices to use for inference."}
    )
    query_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    examples_for_task: str = field(
        default=None, metadata={"help": "Examples for task"}
    )
    examples_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format for examples instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    reranker_name_or_path: str = field(
        default=None, metadata={"help": "The reranker name or path."}
    )
    reranker_peft_path: str = field(
        default=None, metadata={"help": "The reranker peft path."}
    )
    use_bf16: bool = field(
        default=False, metadata={"help": "whether to use bf16 for inference"}
    )
    query_instruction_for_rerank: str = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    passage_instruction_for_rerank: str = field(
        default=None, metadata={"help": "Instruction for passage"}
    )
    passage_instruction_format_for_rerank: str = field(
        default="{}{}", metadata={"help": "Format for passage instruction"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )