import os
import logging
from typing import List
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from src.retrieval import (
    RetrievalArgs, 
)
from .eval_retrieval import main

logger = logging.getLogger(__name__)


@dataclass
class ToolArgs(RetrievalArgs):
    output_dir: str = field(
        default="data/results/tool",
    )
    eval_data: str = field(
        default="llm-embedder:tool/toolbench/test.json",
        metadata={'help': 'Query jsonl.'}
    )
    corpus: str = field(
        default="llm-embedder:tool/toolbench/corpus.json",
        metadata={'help': 'Corpus path for retrieval.'}
    )
    key_template: str = field(
        default="{text}",
        metadata={'help': 'How to concatenate columns in the corpus to form one key?'}
    )

    cutoffs: List[int] = field(
        default_factory=lambda: [1,3,5],
        metadata={'help': 'Cutoffs to evaluate retrieval metrics.'}
    )
    max_neg_num: int = field(
        default=32,
        metadata={'help': 'Maximum negative number to mine.'}
    )
    log_path: str = field(
        default="data/results/tool/toolbench.log",
        metadata={'help': 'Path to the file for logging.'}
    )


if __name__ == "__main__":
    parser = HfArgumentParser([ToolArgs])
    args, = parser.parse_args_into_dataclasses()
    if args.retrieval_method == "dense":
        output_dir = os.path.join(args.output_dir, args.query_encoder.strip(os.sep).replace(os.sep, "--"))
        args.output_dir = output_dir
    else:
        output_dir = os.path.join(args.output_dir, args.retrieval_method)
    main(args)