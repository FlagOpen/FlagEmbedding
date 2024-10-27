from dataclasses import dataclass, field
from typing import Optional, List
from ..retrieval.args import BaseArgs


@dataclass
class LMArgs(BaseArgs):
    model_name_or_path: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    truncation_side: str = field(
        default="right",
        metadata={'help': 'Tokenizer truncation side.'}
    )
    context_max_length: int = field(
        default=2048,
        metadata={'help': 'Evaluation json file.'},
    )
    add_position_ids: bool = field(
        default=False,
        metadata={'help': 'Create position ids based on attention masks? Useful when training left-padded models with absolute position embeddings.'}
    )

    lm_dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    lm_device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    lm_batch_size: int = field(
        default=2,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )

    add_llama_inst: bool = field(
        default=False,
        metadata={'help': 'Add llama2-chat instructions? ([INST] and [/INST])'}
    )


@dataclass
class SRLMArgs(LMArgs):
    context_max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens in total as inputs?'}
    )
    context_window_size: int = field(
        default=2048,
        metadata={'help': 'How many tokens the model can process at the same time?'}   
    )
    target_length: int = field(
        default=1024,
        metadata={'help': 'How many tokens to compute perplexity?'}  
    )
    chunk_size: int = field(
        default=128,
        metadata={'help': 'How many tokens in a chunk?'}
    )
    key_num: int = field(
        default=1,
        metadata={'help': 'How many chunks to retrieve at a time?'}
    )
    chunk_batch_size: int = field(
        default=2,
        metadata={'help': 'How many retrieval & generation to execute in parallel?'}  
    )
    add_key_continuation: bool = field(
        default=False,
        metadata={'help': 'Add continuation as keys?'}
    )
    retrieval_method: str = field(
        default='dense',
        metadata={'help': 'How to retrieve?'}
    )
    order_method: str = field(
        default='sequential',
        metadata={'help': 'How to retrieve?'}
    )
    integrate_method: str = field(
        default="concat",
        metadata={'help': 'How to integrate retrieved chunks. Replace: replace the most distant chunks. Concat: concatenate at the beginning.'}
    )
    add_sep: Optional[List[int]] = field(
        default=None,
        metadata={'help': 'The tokens to add after retrieved chunks. "none" means no sep.'}
    )


@dataclass
class GenerationArgs:
    do_sample: bool = field(
        default=False, 
        metadata={'help': 'Sample when decoding?'}
    )
    num_return_sequences: int = field(
        default=1, 
        metadata={'help': 'How many sequences to generate?'}
    )
    temperature: float = field(
        default=1.0, 
        metadata={'help': 'Temperature for sampling'}
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Top-p sampling value'}
    )
    max_new_tokens: Optional[int] = field(
        default=32, 
        metadata={'help': 'Maximum new token number.'}
    )
    eos_token_id: Optional[int] = field(
        default=None,
        metadata={'help': 'End of sequence token id.'}
    )
    _from_model_config: bool = field(
        default=False, 
        metadata={'help': 'Load generation config from model config?'}
    )
    def __post_init__(self):
        if self.temperature == 0:
            self.temperature = 1e-8