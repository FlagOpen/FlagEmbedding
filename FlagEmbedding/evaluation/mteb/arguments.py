from dataclasses import dataclass, field
from typing import List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class MTEBEvalArgs(AbsEvalArgs):
    """
    Argument class for MTEB evaluation.
    """
    languages: List[str] = field(
        default=None, metadata={"help": "Languages to evaluate. Default: eng"}
    )
    tasks: List[str] = field(
        default=None, metadata={"help": "Tasks to evaluate. Default: None"}
    )
    task_types: List[str] = field(
        default=None, metadata={"help": "The task types to evaluate. Default: None"}
    )
    use_special_instructions: bool = field(
        default=False, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )
    examples_path: str = field(
        default=None, metadata={"help": "Use specific examples in the path. Default: None"}
    )