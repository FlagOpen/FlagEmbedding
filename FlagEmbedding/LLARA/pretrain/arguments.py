import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


def default_list() -> List[int]:
    return ['v_proj', 'q_proj', 'k_proj', 'gate_proj', 'down_proj', 'o_proj', 'up_proj']


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
    # cache_dir: Optional[str] = field(
    #     default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    # )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: List[str] = field(
        default_factory=default_list
    )
    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    )
    token: str = field(
        default=""
    )
    cache_dir: str = field(
        default="./LMs"
    )


@dataclass
class DataArguments:
    cache_path: str = field(
        default='./data_dir'
    )

    train_data: str = field(
        default='./toy_finetune_data.jsonl', metadata={"help": "Path to train data"}
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    cutoff_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    remove_stop_words: bool = field(
        default=False
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class PretrainTrainingArguments(TrainingArguments):
    mask: bool = field(default=True, metadata={"help": "mask the input part"})