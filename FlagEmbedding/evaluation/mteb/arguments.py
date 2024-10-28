from dataclasses import dataclass, field
from typing import List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class MTEBEvalArgs(AbsEvalArgs):
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
    use_special_examples: bool = field(
        default=False, metadata={"help": "Whether to use specific examples in `examples` for evaluation. Default: False"}
    )
