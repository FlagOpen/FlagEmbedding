import os
from dataclasses import dataclass, field
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict


@dataclass
class ModelArgs:
    model_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    data_root: str = field(
        default="/data/activation-beacon", 
        metadata={'help': 'The base directory storing all data used for training and evaluation. If specified, make sure all train_data, eval_data, and corpus are path relative to data_root!'},
    )
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Training json file or glob to match a list of files.'},
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json file.'},
    )
    
    model_name_or_path: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Huggingface access token.'}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={'help': 'Use flash attention?'}
    )

    max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input.'},
    )
    add_chat_inst: bool = field(
        default=False,
        metadata={'help': 'Add chat model instruction?'}
    )

    rope_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to scale RoPE?'},
    )
    rope_factor: float = field(
        default=1.,
        metadata={'help': 'RoPE scaling factor.'},
    )
    
    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )

    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )

    enable_beacon: bool = field(
        default=False,
        metadata={'help': 'Enable activation beacon?'}
    )
    beacon_window: int = field(
        default=1024,
        metadata={'help': 'The initial sliding window size.'}
    )
    beacon_stride: List[int] = field(
        default_factory=lambda: [1024],
        metadata={'help': 'The possible strides of sliding window.'}
    )
    beacon_stride_mix: str = field(
        default="step-random",
        metadata={'help': 'How to determine the beacon_stride for each input. {step-random, instance-random, mix-random}'}
    )
    beacon_attn: str = field(
        default="step-expansion",
        metadata={'help': 'How to assign attention masks of beacon tokens? {segmentation, step-expansion, full-converage}'}
    )
    beacon_attend_previous: bool = field(
        default=True,
        metadata={'help': 'The beacons in the current window can attend to beacons in previous windows?'}
    )
    beacon_ratio: List[int] = field(
        default_factory=lambda: [0,2,4,8,16,32,64,128],
        metadata={'help': 'Condensing ratios for beacons.'}
    )
    beacon_ratio_mix: str = field(
        default="adapt-1024",
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, mix-random, adapt-x}'}
    )
    beacon_param: List[str] = field(
        default_factory=lambda: ['q', 'k', 'v', 'o'],
        metadata={'help': 'The introduced parameters for beacon.'}
    )
    retrieval_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to retrieve? {bm25}'}
    )
    retrieval_topk: int = field(
        default=2,
        metadata={'help': 'How many windows to retrieve?'}
    )

    def resolve_path(self, path):
        """Resolve any path starting with 'activation-beacon:' to relative path against data_root."""
        pattern = "activation-beacon:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        return path

    def __post_init__(self):        
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)
        

@dataclass
class TrainingArgs(TrainingArguments):
    # ==============================
    # Colossal ai specific arguments
    # ==============================
    use_colossal: bool = field(
        default=False,
        metadata={'help': 'Use colossal trainer?'}
    )
    colossal_plugin: str = field(
        default="gemini",
        metadata={'help': 'The plugin name for colossalai.'}
    )
    mixed_precision: str = field(
        default="bf16",
        metadata={'help': 'The mixed precision for colossalai.'}
    )
    save_optimizer: bool = field(
        default=False,
        metadata={'help': 'Save optimizer and scheduler in checkpoint?'}
    )
    
    # ==============================
    # Common arguments
    # ==============================
    output_dir: str = field(
        default="data/outputs/pretrain",
    )

    per_device_train_batch_size: int = field(
        default=1,
        metadata={'help': 'Train batch size.'}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Remove columns in the dataset that are not registered in the forward function?'}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={'help': 'Find unusuable parameters?'}
    )
    # NOTE: essential to keep comuputation graph because we need gradients for beacon tokens
    gradient_checkpointing_kwargs: Optional[Dict] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    report_to: str = field(
        default="none",
        metadata={'help': 'Log results by external tools?'}
    )

    # ==============================
    # Customized arguments
    # ==============================
    min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for training?'}
    )
    max_train_num_per_data: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples at most for each train_data?'}
    )
    
    eval_method: str = field(
        default="perplexity",
        metadata={'help': 'How to evaluate during training? {perplexity, generation}'}
    )
    eval_max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input in evaluation.'},
    )
    eval_min_length: int = field(
        default=512,
        metadata={'help': 'How many tokens at minimum for each input in evaluation.'},
    )
    max_eval_num: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for validation?'}
    )

    lora_tune: bool = field(
        default=False,
        metadata={"help": "Use LoRA fine-tuning?"},
    )
    lora_rank: int = field(
        default=8,
        metadata={'help': 'LoRA rank.'}
    )
    lora_alpha: int = field(
        default=16,
        metadata={'help': 'LoRA scaling factor.'}
    )
    lora_dropout: float = field(
        default=0.,
        metadata={'help': 'LoRA dropout p.'}
    )
    lora_targets: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Module name patterns to add LoRA."},
    )
    lora_extra_params: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    )

    metrics: List[str] = field(
        default_factory=lambda: ["rouge"],
        metadata={'help': 'List of metrics. {rouge, acc}'}
    )
    log_path: str = field(
        default="data/outputs/metrics.log",
        metadata={'help': 'Log file path.'}
    )
