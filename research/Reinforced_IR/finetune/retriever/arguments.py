from dataclasses import dataclass, field
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderDataArguments

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainingArguments,
    AbsEmbedderModelArguments
)


@dataclass
class IREmbedderTrainingArguments(AbsEmbedderTrainingArguments):
    """
    Training argument class for M3.
    """
    use_linear_for_answer: bool = field(default=False, metadata={"help": "use linear fuse for answer"})
    linear_path: str = field(default=None, metadata={"help": "The linear weight path"})
    training_type: str = field(default='retrieval_answer', metadata={"help": "whether to use answer"})
    answer_temperature: float = field(default=None, metadata={"help": "temperature for answer"})
    normalize_answer: bool = field(default=True, metadata={"help": "normalize answer"})
    
@dataclass
class IREmbedderDataArguments(AbsEmbedderDataArguments):
    """
    Data argument class for M3.
    """
    answer_inbatch: bool = field(default=False)
