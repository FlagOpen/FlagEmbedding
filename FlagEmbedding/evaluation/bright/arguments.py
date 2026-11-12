from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class BrightEvalArgs(AbsEvalArgs):
    """
    Argument class for Bright evaluation.
    """
    task_type: str = field(
        default="short", metadata={"help": "The task type to evaluate on. Available options: ['short', 'long']. Default: short", "choices": ["short", "long"]}
    )
    use_special_instructions: bool = field(
        default=True, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: True"}
    )
