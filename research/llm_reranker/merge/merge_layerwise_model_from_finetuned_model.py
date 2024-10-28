from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_layerwise_finetuned_llm(model_name_or_path, lora_name_or_path, save_path, cache_dir: str = None, token: str = None):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
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
