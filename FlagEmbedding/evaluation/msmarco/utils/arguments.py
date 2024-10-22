from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs

@dataclass
class AbsEvalArgs:
    text_type: str = field(
        default='passage', metadata={"help": "The type of text to be searched. Default: passage"}
    )
    splits: str = field(
        default='dev', metadata={"help": "Splits to evaluate. Default: dev. Your choosen: dev, dl19, dl20"}
    )
    