from .utils import FileLogger, DefaultDataCollator, makedirs, split_file_dir_name_ext, clear_dir, get_max_length_in_nested_lists, pad_nested_lists, mask_nested_lists, normalize_text, wrap_text, load_json, save_json, load_pickle, save_pickle, add_eos, remove_eos
from .chat import apply_chat_template
from .args import ModelArgs, TrainingArgs
from .data import Data
from .modeling_utils import evaluate_perplexity, evaluate_generation, move_to_device

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def get_model_and_tokenizer(model_args, accelerator=None, **kwargs):
    """Load model and tokenizer."""
    import torch
    from dataclasses import asdict
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging
    from transformers.integrations import is_deepspeed_zero3_enabled

    from .llama import LlamaForCausalLM, LlamaConfig
    from .mistral import MistralForCausalLM, MistralConfig

    from .args import ModelArgs

    ARCHITECTURE_TO_CLASS = {
        'LlamaForCausalLM': (LlamaConfig, LlamaForCausalLM),
        'MistralForCausalLM': (MistralConfig, MistralForCausalLM),
    }

    logger = logging.get_logger(__name__)

    model_args: ModelArgs

    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)

    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        if accelerator is not None:
            device_map = {"": accelerator.device}
        else:
            device_map = {"": "cpu"}
    
    rope_kwargs = {}
    rope_method = model_args_dict["rope_method"]
    if rope_method is not None:
        rope_scaling = {
            "type": rope_method,
            "factor": model_args_dict["rope_factor"]
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    access_token = model_args_dict["access_token"]

    attn_kwargs = {}
    attn_impl = model_args_dict["attn_impl"]
    if attn_impl is not None:
        attn_kwargs = {"attn_implementation": attn_impl}

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")
    if model_args_dict["enable_beacon"]:
        # use architecture attribute to distinguish different models
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir, token=access_token, trust_remote_code=True)
        architecture = config.architectures[0]
        config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]

        config = config_class.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir,
            token=access_token,
            beacon_window=model_args_dict["beacon_window"],
            beacon_stride=model_args_dict["beacon_stride"],
            beacon_attn=model_args_dict["beacon_attn"],
            beacon_ratio=model_args_dict["beacon_ratio"],
            beacon_ratio_mix=model_args_dict["beacon_ratio_mix"],
            beacon_param=model_args_dict["beacon_param"],
            beacon_sink_size=model_args_dict["beacon_sink_size"],
            retrieval_method=model_args_dict["retrieval_method"],
            retrieval_topk=model_args_dict["retrieval_topk"],
            retrieval_key_length=model_args_dict["retrieval_key_length"],
            retrieval_cache_dir=model_args_dict["model_cache_dir"],
            **rope_kwargs,
            **attn_kwargs,
        )
        model = model_class.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
            trust_remote_code=True,
            # NOTE: do not destroy the default rope_scaling of the model
            **rope_kwargs,
            **attn_kwargs
        )

    # load lora
    if model_args_dict["lora"] is not None:
        from peft import PeftModel
        logger.info(f"loading lora from {model_args_dict['lora']}...")
        model = PeftModel.from_pretrained(model, model_args_dict["lora"])
        if model_args_dict["lora_unload"]:
            model = model.merge_and_unload()
    
    if accelerator is not None:
        model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True).eval()
    else:
        model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, padding_side=model_args_dict["padding_side"], token=access_token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # override the default generation config
    generation_config = model_args.get_generation_config()
    if len(generation_config):
        logger.info(f"Overriding the model's default generation config with {generation_config}.")
        unused_config = model.generation_config.update(**generation_config)
        if len(unused_config):
            logger.warning(f"The following attributes are not used when overriding the generation configurations: {unused_config}")

    return model, tokenizer

