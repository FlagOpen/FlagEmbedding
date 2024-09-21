from typing import Optional, List
from dataclasses import dataclass, field

from src.abc.finetune.embedder import AbsTrainingArguments, AbsModelArguments


@dataclass
class M3ModelArguments(AbsModelArguments):
    colbert_dim: int = field(default=-1, metadata={"help": "Dim of colbert linear"})


@dataclass
class M3TrainingArguments(AbsTrainingArguments):
    unified_finetuning: bool = field(default=False, metadata={"help": "use unify fine-tuning"})
    use_self_distill: bool = field(default=False, metadata={"help": "use self-distill when using unify fine-tuning"})
    fix_encoder: bool = field(default=False, metadata={"help": "Freeze the parameters of encoder"})
    self_distill_start_step: int = field(default=-1, metadata={"help": "Num of step when using self-distill"})
