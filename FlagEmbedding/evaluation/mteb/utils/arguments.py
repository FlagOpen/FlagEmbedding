from dataclasses import dataclass, field
from typing import List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs

@dataclass
class MTEBEvalArgs(AbsEvalArgs):
    task_types: List[str] = field(
        default=None, metadata={"help": "The tasks to evaluate. Default: None"}
    )