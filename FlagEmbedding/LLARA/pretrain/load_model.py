
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from modeling import PreLlamaModel

def get_model(model_args, use_gradient_checkpointing: bool = False):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            token=model_args.token,
                                            cache_dir=model_args.cache_dir,
                                            )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            token=model_args.token,
                                            cache_dir=model_args.cache_dir,
                                            )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    if use_gradient_checkpointing:
        config.use_cache = False

    if model_args.model_name_or_path:
        model = PreLlamaModel.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            attn_implementation='sdpa',
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        print("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=model_args.lora_rank,
            target_modules=model_args.target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model