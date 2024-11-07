import os
import re
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from FlagEmbedding.finetune.reranker.decoder_only.base.arguments import RerankerModelArguments

logger = logging.getLogger(__name__)


def find_largest_checkpoint(checkpoint_dir):
    """Find the largest checkpoint from directory.

    Args:
        checkpoint_dir (str): Directory to the checkpoint.

    Returns:
        str: Directory to the checkpoint, None no matching found.
    """
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


def get_model(model_args: RerankerModelArguments):
    """Get the model.

    Args:
        model_args (RerankerModelArguments): Model arguments instance.

    Returns:
        transformers.PreTrainedModel or PeftModel: The loaded model.
    """
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            trust_remote_code=model_args.trust_remote_code,
            token=model_args.token,
            cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            token=model_args.token,
            cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.use_cache = False

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # torch_dtype=torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.raw_peft is not None:
        for peft_path in model_args.raw_peft:
            model = PeftModel.from_pretrained(model, peft_path)
            model = model.merge_and_unload()

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
                modules_to_save=model_args.modules_to_save,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    return model


def save_merged_model(model_args: RerankerModelArguments, output_dir: str):
    """
    Loads and save a model with specified configurations, merges it with PEFT layers if available.

    Args:
        model_args (RerankerModelArguments): Model arguments instance.
        output_dir (str): Directory to save the model.
    """
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    config.use_cache = False

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            # torch_dtype=torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            trust_remote_code=model_args.trust_remote_code,
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_args.from_config(config)

    if model_args.raw_peft is not None:
        for peft_path in model_args.raw_peft:
            model = PeftModel.from_pretrained(model, peft_path)
            model = model.merge_and_unload()

    try:
        model = PeftModel.from_pretrained(model, output_dir)
        model = model.merge_and_unload()
    except:
        model = PeftModel.from_pretrained(model, find_largest_checkpoint(output_dir))
        model = model.merge_and_unload()

    model.save_pretrained(os.path.join(output_dir, 'merged_model'))

    try:
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained(find_largest_checkpoint(output_dir))

    tokenizer.save_pretrained(os.path.join(output_dir, 'merged_model'))
