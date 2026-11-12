from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .configuration_minicpm_reranker import LayerWiseMiniCPMConfig


def merge_layerwise_raw_llm(model_name_or_path, lora_name_or_path, save_path, cache_dir: str = None, token: str = None):
    config = AutoConfig.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise',
                                        cache_dir=cache_dir,
                                        token=token,
                                        trust_remote_code=True)
    train_config = LayerWiseMiniCPMConfig.from_pretrained(lora_name_or_path)
    config.attention_bias = train_config.attention_bias
    config.attention_dropout = train_config.attention_dropout
    config.bos_token_id = train_config.bos_token_id
    config.dim_model_base = train_config.dim_model_base
    config.eos_token_id = train_config.eos_token_id
    config.head_multi = train_config.head_multi
    config.head_type = train_config.head_type
    config.hidden_act = train_config.hidden_act
    config.hidden_size = train_config.hidden_size
    config.initializer_range = train_config.initializer_range
    config.max_position_embeddings = train_config.max_position_embeddings
    config.model_type = train_config.model_type
    config.num_attention_heads = train_config.num_attention_heads
    config.num_hidden_layers = train_config.num_hidden_layers
    config.num_key_value_heads = train_config.num_key_value_heads
    config.pretraining_tp = train_config.pretraining_tp
    config.rms_norm_eps = train_config.rms_norm_eps
    config.rope_scaling = train_config.rope_scaling
    config.rope_theta = train_config.rope_theta
    config.scale_depth = train_config.scale_depth
    config.scale_emb = train_config.scale_emb
    config.start_layer = train_config.start_layer
    config.transformers_version = train_config.transformers_version
    config.use_cache = train_config.use_cache
    config.vocab_size = train_config.vocab_size

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                        config=config,
                                                        cache_dir=cache_dir,
                                                        token=token,
                                                        trust_remote_code=True)

    model = PeftModel.from_pretrained(model, lora_name_or_path)
    model = model.merge_and_unload()
    model.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_name_or_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  cache_dir=cache_dir,
                                                  token=token,
                                                  trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        if 'mistral' in model_name_or_path.lower():
            tokenizer.padding_side = 'left'
    tokenizer.save_pretrained(save_path)
