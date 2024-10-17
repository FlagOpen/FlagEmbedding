from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderTrainingArguments,
    AbsEmbedderModelArguments
)


@dataclass
class EncoderOnlyEmbedderM3ModelArguments(AbsEmbedderModelArguments):
    colbert_dim: int = field(default=-1, metadata={"help": "Dim of colbert linear"})


@dataclass
class EncoderOnlyEmbedderM3TrainingArguments(AbsEmbedderTrainingArguments):
    unified_finetuning: bool = field(default=False, metadata={"help": "use unify fine-tuning"})
    use_self_distill: bool = field(default=False, metadata={"help": "use self-distill when using unify fine-tuning"})
    fix_encoder: bool = field(default=False, metadata={"help": "Freeze the parameters of encoder"})
    self_distill_start_step: int = field(default=-1, metadata={"help": "Num of step when using self-distill"})
