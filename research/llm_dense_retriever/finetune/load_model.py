import os
import re

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

def find_largest_checkpoint(checkpoint_dir):
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    max_number = -1
    max_checkpoint_file = None
    for file in os.listdir(checkpoint_dir):
        match = checkpoint_pattern.search(file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_checkpoint_file = file
    if max_checkpoint_file:
        return os.path.join(checkpoint_dir, max_checkpoint_file)
    else:
        return None

def get_model(model_args, output_dir, resize, resize_tokens):

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
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            # torch_dtype=torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        print("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.raw_peft is not None:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, 'embedding', 'emb.pth')))
        model = PeftModel.from_pretrained(model, model_args.raw_peft)
        model = model.merge_and_unload()

    if resize:
        model.resize_token_embeddings(resize_tokens)
        os.makedirs(os.path.join(output_dir, 'embedding'), exist_ok=True)
        torch.save(model.embed_tokens, os.path.join(output_dir, 'embedding', 'emb.pth'))
        target_modules = model_args.target_modules
    else:
        target_modules = model_args.target_modules
        if 'embed_tokens' in target_modules:
            target_modules.remove('embed_tokens')

    if model_args.from_peft is not None:
        if os.path.exists(os.path.join(model_args.from_peft, 'embedding')):
            model.set_input_embeddings(torch.load(os.path.join(model_args.from_peft, 'embedding', 'emb.pth')))
            torch.save(model.embed_tokens, os.path.join(output_dir, 'embedding', 'emb.pth'))
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=target_modules,
                modules_to_save=model_args.modules_to_save,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    return model

def save_merged_model(model_args, output_dir):
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
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            # torch_dtype=torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        print("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.raw_peft is not None:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, 'embedding', 'emb.pth')))
        model = PeftModel.from_pretrained(model, model_args.raw_peft)
        model = model.merge_and_unload()

    if os.path.exists(os.path.join(output_dir, 'embedding', 'emb.pth')):
        model.set_input_embeddings(torch.load(os.path.join(output_dir, 'embedding', 'emb.pth')))

    try:
        model = PeftModel.from_pretrained(model, output_dir)
        model = model.merge_and_unload()
    except:
        model = PeftModel.from_pretrained(model, find_largest_checkpoint(output_dir))
        model = model.merge_and_unload()

    model.save_pretrained(os.path.join(output_dir, 'full_model'))

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    tokenizer.save_pretrained(os.path.join(output_dir, 'full_model'))