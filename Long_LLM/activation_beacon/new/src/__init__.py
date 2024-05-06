from .utils import FileLogger, DefaultDataCollator, makedirs, split_file_dir_name_ext, clear_dir, get_max_length_in_nested_lists, pad_nested_lists, mask_nested_lists, normalize_text, wrap_text, load_json, save_json, load_pickle, save_pickle, add_eos, remove_eos, format_numel_str
from .chat import apply_chat_template
from .args import ModelArgs
from .data import Data
from .modeling_utils import evaluate_perplexity, evaluate_generation, move_to_device

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def get_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):
    """Load model and tokenizer."""
    import torch
    import transformers
    from dataclasses import asdict
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging
    from transformers.integrations import is_deepspeed_zero3_enabled
    from packaging import version

    from .args import ModelArgs

    logger = logging.get_logger(__name__)

    model_args: ModelArgs

    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)
    
    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    access_token = model_args_dict["access_token"]

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

    tokenizer_kwargs = {}
    if model_args_dict["no_use_fast"]:
        tokenizer_kwargs = {"use_fast": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        padding_side=model_args_dict["padding_side"], 
        token=access_token, 
        trust_remote_code=True,
        **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if return_tokenizer_only:
        return tokenizer

    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}
    
    rope_kwargs = {}
    rope_theta = model_args_dict["rope_theta"]
    if rope_theta is not None:
        rope_kwargs["rope_theta"] = rope_theta
    rope_method = model_args_dict["rope_method"]
    if rope_method is not None:
        rope_factor = model_args_dict["rope_factor"]
        rope_scaling = {
            "type": rope_method,
            "factor": rope_factor
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    attn_kwargs = {}
    attn_impl = model_args_dict["attn_impl"]
    if attn_impl is not None:
        if version.parse(transformers.__version__) <= version.parse("4.36"):
            if attn_impl == "flash_attention_2":
                attn_kwargs["use_flash_attention_2"] = True
        else:
            attn_kwargs["attn_implementation"] = attn_impl

    # from_pretrained_kwargs = {}
    # if attn_impl == "flash_attention_2" and version.parse(transformers.__version__) <= version.parse("4.36"):
    #     from_pretrained_kwargs["use_flash_attention_2"] = True

    beacon_kwargs = {}
    for k, v in model_args_dict.items():
        if k.startswith("beacon") and v is not None:
            beacon_kwargs[k] = v
        elif k.startswith("retrieval") and v is not None:
            beacon_kwargs[k] = v

    # use architecture attribute to distinguish different models
    probe_config = AutoConfig.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        token=access_token, 
        trust_remote_code=True
    )
    architecture = probe_config.architectures[0]

    extra_kwargs = {}
    if model_args_dict["max_position_embeddings"] is not None:
        extra_kwargs["max_position_embeddings"] = model_args_dict["max_position_embeddings"]
    if architecture == "MistralForCausalLM" and model_args_dict["mistral_sliding_window"] is not None:
        extra_kwargs["sliding_window"] = model_args_dict["mistral_sliding_window"]

    if model_args_dict["enable_beacon"]:
        from .llama import LlamaForCausalLM, LlamaConfig
        from .mistral import MistralForCausalLM, MistralConfig
        ARCHITECTURE_TO_CLASS = {
            'LlamaForCausalLM': (LlamaConfig, LlamaForCausalLM),
            'MistralForCausalLM': (MistralConfig, MistralForCausalLM),
        }

        config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]

        config = config_class.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir,
            token=access_token,
            **beacon_kwargs,
            **rope_kwargs,
            **attn_kwargs,
            **extra_kwargs,
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
        # NOTE: mistral models do not have Linear/NTK interpolation, we should replace the default RoPE with the proper one
        # if architecture == "MistralForCausalLM":
        #     if rope_method is not None:
        #         import inspect
        #         import transformers
        #         logger.info(f"replacing RoPE module (type: {rope_method} factor: {rope_factor}) for mistral models...")
        #         if rope_method == "linear":
        #             from .mistral.modeling_mistral import MistralLinearScalingRotaryEmbedding as MistralRotaryEmbedding
        #         elif rope_method == "dynamic":
        #             from .mistral.modeling_mistral import MistralDynamicNTKScalingRotaryEmbedding as MistralRotaryEmbedding
        #         else:
        #             raise ValueError(f"Make sure the rope method is either linear or dynamic, found {rope_method}!")
        #         default_args = inspect.getfullargspec(MistralRotaryEmbedding.__init__).args
        #         assert default_args[-1] == "scaling_factor"
        #         default_arg_values = list(MistralRotaryEmbedding.__init__.__defaults__)
        #         default_arg_values[-1] = rope_factor
        #         MistralRotaryEmbedding.__init__.__defaults__ = tuple(default_arg_values)
        #         # override the original rope module
        #         transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = MistralRotaryEmbedding
        #         del rope_kwargs["rope_scaling"]

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
            trust_remote_code=True,

            # NOTE: do not destroy the default rope_scaling of the model
            **rope_kwargs,
            **attn_kwargs,
            **extra_kwargs,
        )

        logger.info(model.config)

    # load lora
    if model_args_dict["lora"] is not None:
        from peft import PeftModel
        logger.info(f"loading lora from {model_args_dict['lora']}...")
        model = PeftModel.from_pretrained(
            model, 
            model_args_dict["lora"],
            torch_dtype=dtype,
            device_map=device_map,
        )
        if model_args_dict["lora_unload"]:
            model = model.merge_and_unload()

    if model_args_dict["enable_tp"]:
        import tensor_parallel as tp
        logger.info("enabling tensor parallelism...")
        
        # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
        model = tp.tensor_parallel(model, sharded=True)

        # if distributed, return would be a tuple
        if isinstance(model, tuple):
            model = model[0]

    model = model.eval()

    if evaluation_mode:
        # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
        model.requires_grad_(False)

    # override the default generation config
    generation_config = model_args.get_generation_config()
    if len(generation_config):
        logger.info(f"Overriding the model's default generation config with {generation_config}.")
        unused_config = model.generation_config.update(**generation_config)
        if len(unused_config):
            logger.warning(f"The following attributes are not used when overriding the generation configurations: {unused_config}")

    return model, tokenizer

