import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    knowledge_distillation: bool = field(
        default=False, metadata={"help": "Use knowledge distillation when `pos_scores` and `neg_scores` are in features of training data"}
    )
    train_data: str = field(
        default=None, metadata={"help": "One or more paths to training data", "nargs": "+"}
    )
    cache_path: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the cached data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=None, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )
    
    same_task_within_batch: bool = field(
            default=False, metadata={"help": "All samples in the same batch comes from the same task."}
    )
    shuffle_ratio: float = field(
            default=0.0, metadata={"help": "The ratio of shuffling the text"}
    )
    
    small_threshold: int = field(
            default=0, metadata={"help": "The threshold of small dataset. All small dataset in the same directory will be merged into one dataset."}
    )
    drop_threshold: int = field(
            default=0, metadata={"help": "The threshold for dropping merged small dataset. If the number of examples in the merged small dataset is less than this threshold, it will be dropped."}
    )

    def __post_init__(self):
        for train_dir in self.train_data:
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"cannot find file: {train_dir}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    enable_sub_batch: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})
    
    unified_finetuning: bool = field(default=False, metadata={"help": "use unify fine-tuning"})
    use_self_distill: bool = field(default=False, metadata={"help": "use self-distill when using unify fine-tuning"})
    fix_encoder: bool = field(default=False, metadata={"help": "Freeze the parameters of encoder"})
    colbert_dim: int = field(default=-1, metadata={"help": "Dim of colbert linear"})
    self_distill_start_step: int = field(default=-1, metadata={"help": "Num of step when using self-distill"})
