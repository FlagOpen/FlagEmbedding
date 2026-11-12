import copy

import torch
from torch import nn

from mistral_model import CostWiseMistralForCausalLM, CostWiseHead
from mistral_config import CostWiseMistralConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


def get_model(model_args, training_args, output_token_id):
    config = CostWiseMistralConfig.from_pretrained(model_args.model_name_or_path,
                                                 token=model_args.token,
                                                 cache_dir=model_args.cache_dir,
                                                 trust_remote_code=True)
    if model_args.use_flash_attn:
        model = CostWiseMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            use_flash_attention_2=True,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            trust_remote_code=True,
            config=config
        )
    else:
        model = CostWiseMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention_2=False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            trust_remote_code=True,
            config=config
        )
    model.config.use_cache = False
    if model_args.layer_wise:
        lm_head = nn.ModuleList([CostWiseHead(
            model.config.hidden_size, 1) for _ in range(
            model_args.start_layer,
            model.config.num_hidden_layers + 1,
            model_args.layer_sep)])
        state_dict_back = model.lm_head.state_dict()
        state_dict_back['weight'] = state_dict_back['weight'][output_token_id: output_token_id + 1, :]
        for i in range(len(lm_head)):
            lm_head[i].linear_head.load_state_dict(state_dict_back)
        model.set_output_embeddings(lm_head)
        model.config.start_layer = model_args.start_layer
        model.config.layer_sep = model_args.layer_sep
        model.config.layer_wise = model_args.layer_wise

    if model_args.raw_peft is not None:
        for raw_peft in model_args.raw_peft:
            model = PeftModel.from_pretrained(model, raw_peft)
            model = model.merge_and_unload()

    tmp_model = None
    if model_args.use_lora:
        # if model_args.finetune_type == 'layer':
        #     peft_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         inference_mode=False,
        #         r=model_args.lora_rank,
        #         target_modules=model_args.target_modules,
        #         lora_alpha=model_args.lora_alpha,
        #         lora_dropout=model_args.lora_dropout,
        #         modules_to_save=model_args.lora_extra_parameters
        #     )

        # else:
        #     peft_config = LoraConfig(
        #         task_type=TaskType.CAUSAL_LM,
        #         inference_mode=False,
        #         r=model_args.lora_rank,
        #         target_modules=model_args.target_modules,
        #         lora_alpha=model_args.lora_alpha,
        #         lora_dropout=model_args.lora_dropout,
        #         modules_to_save=model_args.lora_extra_parameters,
        #         layers_to_transform=model_args.compress_layers
        #     )
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

    return model, tmp_model