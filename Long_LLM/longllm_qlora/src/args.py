import os
import json
from dataclasses import dataclass, field, asdict
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
        default="/data/long-llm", 
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
    no_use_fast: bool = field(
        default=False,
        metadata={'help': 'Do not use fast tokenizer?'}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: Optional[str] = field(
        default="sdpa",
        metadata={'help': 'The implementation of attention.'}
    )

    max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input.'},
    )
    chat_template: str = field(
        default="llama-3",
        metadata={'help': 'Instruction template name in fastchat.'}
    )

    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum position.'},
    )
    mistral_sliding_window: Optional[int] = field(
        default=None,
        metadata={'help': 'Sliding window size in Mistral models.'},
    )
    rope_theta: Optional[float] = field(
        default=None,
        metadata={'help': 'RoPE base (theta).'},
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
    load_in_4_bit: bool = field(
        default=False,
        metadata={'help': 'Load model in 4 bits?'},
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

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )

    enable_beacon: bool = field(
        default=False,
        metadata={'help': 'Enable activation beacon?'}
    )
    beacon_window: Optional[int] = field(
        default=None,
        metadata={'help': 'The initial sliding window size.'}
    )
    beacon_stride: Optional[int] = field(
        default=None,
        metadata={'help': 'The stride of the sliding window.'}
    )
    beacon_attn: Optional[str] = field(
        default=None,
        metadata={'help': 'How to assign attention masks of beacon tokens? {segmentation, step-expansion, full-converage}'}
    )
    beacon_ratio: Optional[List[int]] = field(
        default=None,
        metadata={'help': 'Condensing ratios for beacons.'}
    )
    beacon_ratio_mix: Optional[str] = field(
        default=None,
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    beacon_param: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'The introduced parameters for beacon.'}
    )
    beacon_embed_init: str = field(
        default="eos",
        metadata={'help': 'Initialize beacon embedding from eos/bos embedding.'}
    )
    beacon_sink_size: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of activations that are always kept in the head of the sequence according to StreamingLLM.'}
    )
    beacon_attend_prev: Optional[bool] = field(
        default=None,
        metadata={'help': 'Can beacon tokens attend to previous beacon tokens?'}
    )
    retrieval_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to retrieve? {bm25}'}
    )
    retrieval_topk: Optional[int] = field(
        default=None,
        metadata={'help': 'How many windows to retrieve?'}
    )
    retrieval_key_length: Optional[int] = field(
        default=None,
        metadata={'help': 'The key sequence length in retrieval.'}
    )

    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    do_sample: Optional[bool] = field(
        default=None,
        metadata={'help': 'Do sampling when decoding?'},
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )

    def resolve_path(self, path):
        """Resolve any path starting with 'long-llm:' to relative path against data_root."""
        pattern = "long-llm:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        return path
    
    def get_generation_config(self):
        generation_config = {}
        if self.max_new_tokens is not None:
            generation_config["max_new_tokens"] = self.max_new_tokens
        if self.do_sample is not None:
            generation_config["do_sample"] = self.do_sample
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["top_p"] = self.top_p
        return generation_config

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    def __post_init__(self):
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)

        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.output_dir = self.resolve_path(self.output_dir)

        if hasattr(self, "result_dir"):
            if self.result_dir is None: 
                if self.lora is not None:
                    name_or_path_components = [x for x in self.lora.split("/") if len(x)][-2:]
                else:
                    name_or_path_components = [x for x in self.model_name_or_path.split("/") if len(x)][-2:]
                self.result_dir = os.path.join(*name_or_path_components)
            else:
                self.result_dir = self.resolve_path(self.result_dir)


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
    colossal_mp: str = field(
        default="bf16",
        metadata={'help': 'The mixed precision for colossalai.'}
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
    use_reentrant: Optional[bool] = field(
        default=None,
        metadata={'help': "Use reetrant in gradient checkpointing?"}
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

    group_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Group the training data instances by the number of strides in the beacon model. {relaxed, strict}'}
    )
    sort_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Sort the training data instances by the number of strides in the beacon model. {ascend, descend}'}
    )
    only_train_beacon: bool = field(
        default=True,
        metadata={'help': 'Freeze LLM parameters when training beacon parameters?'}
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
    eval_beacon_ratio: List[int] = field(
        default_factory=lambda: [32],
        metadata={'help': 'Condensing ratios for beacons in evaluation.'}
    )
    eval_beacon_ratio_mix: str = field(
        default="adapt-1024",
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, adapt-x}'}
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
        default=32,
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
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Module name patterns to add LoRA."},
    )
    lora_extra_params: List[str] = field(
        default_factory=lambda: ["embed_tokens", "norm"],
        metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    )

    metrics: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'List of metrics. {rouge, save_result}'}
    )
    log_path: str = field(
        default="data/outputs/metrics.log",
        metadata={'help': 'Log file path.'}
    )


    def __post_init__(self):
        if self.use_reentrant is not None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": self.use_reentrant}
        return super().__post_init__()
