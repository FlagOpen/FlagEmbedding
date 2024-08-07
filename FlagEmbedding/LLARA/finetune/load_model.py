import sys

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


def get_model(model_args):
    # if model_args.use_flash_attn:
    #     from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn()

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
    config.use_cache = False

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # load_in_8bit=True,
            # torch_dtype=torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            # low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            # device_map="auto",
        )
    else:
        print("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.from_peft is not None:
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=model_args.target_modules,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout
            )
            model = get_peft_model(model, peft_config)
            # print(model.model.layers[0].self_attn.q_proj.weight.dtype)
            # print(model.model.layers[0].self_attn.q_proj.lora_A.default.weight.dtype)
            # sys.exit(0)
            model.print_trainable_parameters()

    return model