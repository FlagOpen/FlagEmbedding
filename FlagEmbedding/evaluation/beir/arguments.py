from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs


@dataclass
class BEIREvalArgs(AbsEvalArgs):
    """
    Argument class for BEIR evaluation.
    """
    use_special_instructions: bool = field(
        default=False, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )
