import torch
from torch import nn
from transformers import AutoConfig
from .modeling_minicpm_reranker import LayerWiseMiniCPMForCausalLM, LayerWiseHead
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


def get_model(model_args, training_args, only_for_one_logit: int = None):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    if model_args.finetune_type == 'from_raw_model':
        config.use_cache = False
        config.start_layer = config.num_hidden_layers
        config.head_multi = False
        config.head_type = 'raw'

        model = LayerWiseMiniCPMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            trust_remote_code=True,
        )

        config.start_layer = model_args.start_layer
        config.head_multi = model_args.head_multi
        config.head_type = model_args.head_type
        model.config = config

        if model.config.head_type == 'complex':
            if model.config.head_multi == True:
                lm_head = nn.ModuleList([LayerWiseHead(
                    model.config.hidden_size, model.config.vocab_size) for _ in range(
                    model.config.start_layer,
                    model.config.num_hidden_layers + 1)])
                for i in range(len(lm_head)):
                    lm_head[i].linear_head.load_state_dict(model.lm_head.state_dict())
                model.set_output_embeddings(lm_head)
            else:
                lm_head = LayerWiseHead(model.config.hidden_size, 1)
                state_dict_back = model.lm_head.state_dict()
                state_dict_back['weight'] = state_dict_back['weight'][only_for_one_logit: only_for_one_logit + 1, :]
                lm_head.linear_head.load_state_dict(state_dict_back)
                model.set_output_embeddings(lm_head)
        else:
            if only_for_one_logit is None:
                raise ValueError('`only for one logit` cannot be None.')
            if model.config.head_multi == True:
                lm_head = nn.ModuleList([LayerWiseHead(
                    model.config.hidden_size, 1) for _ in range(
                    model.config.start_layer,
                    model.config.num_hidden_layers + 1)])
                state_dict_back = model.lm_head.state_dict()
                state_dict_back['weight'] = state_dict_back['weight'][only_for_one_logit: only_for_one_logit + 1, :]
                for i in range(len(lm_head)):
                    lm_head[i].linear_head.load_state_dict(state_dict_back)
                model.set_output_embeddings(lm_head)
            else:
                lm_head = LayerWiseHead(model.config.hidden_size, 1)
                state_dict_back = model.lm_head.state_dict()
                state_dict_back['weight'] = state_dict_back['weight'][only_for_one_logit: only_for_one_logit + 1, :]
                lm_head.linear_head.load_state_dict(state_dict_back)
                model.set_output_embeddings(lm_head)
        lora_extra_parameters = model_args.lora_extra_parameters
        target_modules = model_args.target_modules
    else:
        config.use_cache = False

        model = LayerWiseMiniCPMForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            trust_remote_code=True,
        )
        target_modules = model_args.target_modules
        target_modules.extend(model_args.lora_extra_parameters)
        lora_extra_parameters = None

    if model_args.from_peft is not None:
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
        model.print_trainable_parameters()
    else:
        if model_args.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_rank,
                target_modules=target_modules,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                modules_to_save=lora_extra_parameters,
            )
            print(peft_config)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    print(model)
    return model