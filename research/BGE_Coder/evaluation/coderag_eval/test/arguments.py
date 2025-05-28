from typing import List
from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation import (
    AbsEvalModelArgs as CodeRAGEvalModelArgs,
)

@dataclass
class CodeRAGEvalArgs:
    dataset: str = field(
        default='humaneval',
        metadata={
            "help": "Task to evaluate. Default: humaneval. Available tasks: "
                    "['humaneval', 'mbpp', 'live_code_bench', 'ds1000', 'odex', 'repoeval_repo', 'swebench_repo', 'code_search_net']",
        }
    )
    max_length: int = field(
        default=2048, metadata={"help": "Max length to use for evaluation."}
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for evaluation."}
    )
    output_file: str = field(
        default="outputs.json",
        metadata={
            "help": "Specify the filepath if you want to save the retrieval (evaluation) results."
        },
    )
    results_file: str = field(
        default="results.json",
        metadata={
            "help": "Specify the filepath if you want to save the retrieval results."
        },
    )