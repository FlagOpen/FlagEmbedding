from .modeling_llama import evaluate_generation, evaluate_perplexity, move_to_device


def get_model_and_tokenizer(model_args, accelerator=None, **kwargs):
    """Load model and tokenizer. Possibly load LoRA for the model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.utils import logging
    from dataclasses import asdict
    from ..args import ModelArgs
    from .configuration_llama import LlamaConfig
    from .modeling_llama import LlamaForCausalLM, is_deepspeed_zero3_enabled

    logger = logging.get_logger(__name__)

    model_args: ModelArgs

    model_args = asdict(model_args)
    model_args.update(**kwargs)

    dtype = model_args["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        if accelerator is not None:
            device_map = {"": accelerator.device}
        else:
            device_map = {"": "cpu"}
    
    rope_kwargs = {}
    rope_method = model_args["rope_method"]
    if rope_method is not None:
        rope_scaling = {
            "type": rope_method,
            "factor": model_args["rope_factor"]
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    model_name_or_path = model_args["model_name_or_path"]
    cache_dir = model_args["model_cache_dir"]
    access_token = model_args["access_token"]
    use_flash_attention_2 = model_args["use_flash_attention_2"]

    logger.info(f"Loading checkpoint from {model_name_or_path}...")

    if model_args["enable_beacon"]:
        config = LlamaConfig.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir,
            token=access_token,
            beacon_window=model_args["beacon_window"],
            beacon_stride=model_args["beacon_stride"],
            beacon_stride_mix=model_args["beacon_stride_mix"],
            beacon_attn=model_args["beacon_attn"],
            beacon_attend_previous=model_args["beacon_attend_previous"],
            beacon_ratio=model_args["beacon_ratio"],
            beacon_ratio_mix=model_args["beacon_ratio_mix"],
            beacon_param=model_args["beacon_param"],
            retrieval_method=model_args["retrieval_method"],
            retrieval_topk=model_args["retrieval_topk"],
            **rope_kwargs,
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path, 
            config=config,
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
            use_flash_attention_2=use_flash_attention_2,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map=device_map, 
            token=access_token,
            use_flash_attention_2=use_flash_attention_2,
            trust_remote_code=True,
            # NOTE: do not destroy the default rope_scaling of the model
            **rope_kwargs,
        )

    # load lora
    if model_args["lora"] is not None:
        from peft import PeftModel
        logger.info(f"loading lora from {model_args['lora']}...")
        model = PeftModel.from_pretrained(model, model_args["lora"])
        if model_args["lora_unload"]:
            model = model.merge_and_unload()
    
    if accelerator is not None:
        model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)
    else:
        model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, padding_side=model_args["padding_side"], token=access_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # NOTE: for models like Qwen, there is no pre-defined eos tokens
        if tokenizer.eos_token is None:
            pad_token = "<|endoftext|>"
        else:
            pad_token = tokenizer.eos_token
        tokenizer.pad_token = pad_token
    
    return model, tokenizer

