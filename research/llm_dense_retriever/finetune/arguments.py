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
    peft_model_path: str = field(
        default=''
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
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
    token: str = field(
        default=".."
    )
    cache_dir: str = field(
        default="../LMs"
    )
    from_peft: str = field(
        default=None
    )
    modules_to_save: str = field(
        default=None
    )
    raw_peft: str = field(
        default=None
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default='cfli/bge-e5data',
        metadata={"help": "Path to train data"}
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
        default=10000000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str = field(
        default="query: ", metadata={"help": "query: "}
    )
    passage_instruction_for_retrieval: str = field(
        default="passage: ", metadata={"help": "passage: "}
    )

    cache_path: str = field(
        default='./data_dir'
    )

    load_from_disk: bool = field(
        default=False, metadata={"help": " whether load the data from disk"}
    )

    load_disk_path: str = field(
        default=None, metadata={"help": " the path to load the data", "nargs": "+"}
    )

    save_to_disk: bool = field(
        default=False, metadata={"help": " whether save the data to disk"}
    )

    save_disk_path: str = field(
        default=None, metadata={"help": " the path to save the data"}
    )

    num_shards: int = field(
        default=0, metadata={
            "help": "number of shards to write, prior than `save_max_shard_size`, default depends on `save_max_shard_size`"}
    )

    save_max_shard_size: str = field(
        default="50GB", metadata={"help": "the max size of the shard"}
    )

    exit_after_save: bool = field(
        default=False, metadata={"help": " whether exit after save the data"}
    )

    shuffle_ratio: float = field(
        default=0.0, metadata={"help": "The ratio of shuffling the text"}
    )
    use_special_tokens: bool = field(default=True)
    nli_all_prompt: bool = field(default=True)
    symmetric_batch_size: int = field(default=128)
    symmetric_train_group_size: int = field(default=8)
    max_class_neg: int = field(default=1000)
    example_query_max_len: int = field(default=64)
    example_passage_max_len: int = field(default=96)
    retrieval_use_examples: bool = field(default=True)

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    sub_batch_size: int = field(default=None)
    cache_chunk_size: int = field(default=-1, metadata={"help": "用于缓存每一步的执行."})