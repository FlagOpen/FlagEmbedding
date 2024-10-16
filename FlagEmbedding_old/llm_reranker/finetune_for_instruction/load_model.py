import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


def get_model(model_args, training_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
        use_flash_attention_2=True if model_args.use_flash_attn else False,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if model_args.from_peft is not None:
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=model_args.target_modules,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                modules_to_save=model_args.lora_extra_parameters
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    print(model)
    return model