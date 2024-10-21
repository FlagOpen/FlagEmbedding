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
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for datasets."}
    )
    overwrite: bool = field(
        default=False, metadata={"help": "whether to overwrite evaluation results"}
    )
