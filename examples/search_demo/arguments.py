from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='BAAI/bge-large-zh-noinstruct',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default='./data', metadata={"help": "Path to wikipedia-22-12"}
    )
