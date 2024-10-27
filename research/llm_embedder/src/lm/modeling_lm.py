import torch
import logging
from tqdm import tqdm
from accelerate import Accelerator
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

logger = logging.getLogger(__name__)


class LM(torch.nn.Module):
    def __init__(self, model_name_or_path=None, padding_side="left", dtype="bf16", cache_dir="/share/LMs", device_map=None, accelerator: Accelerator=None, generation_args: Dict=None) -> None:
        super().__init__()

        logger.info(f"loading tokenizer and model from {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, padding_side=padding_side, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # NOTE: for models like Qwen, there is no pre-defined eos tokens
            if tokenizer.eos_token is None:
                pad_token = "<|endoftext|>"
            else:
                pad_token = tokenizer.eos_token
            tokenizer.pad_token = pad_token

        self.tokenizer = tokenizer
        
        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.accelerator = accelerator

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, torch_dtype=dtype, trust_remote_code=True, device_map=device_map)
        except ValueError:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, torch_dtype=dtype, trust_remote_code=True, device_map=device_map)

        # if device_map is specified, we don't need to move the model to any specific gpu
        if device_map is None:
            if accelerator is not None:
                device = accelerator.device
            else:
                device = torch.device("cpu")
            self.model.to(device)

        # update the model's default generation config
        if generation_args is not None:
            generation_config = self.model.generation_config.to_dict()
            generation_config.update(generation_args)
            generation_config.update({
                "pad_token_id": self.tokenizer.pad_token_id
            })
            self.model.generation_config = GenerationConfig(**generation_config)

    @property
    def device(self):
        if self.accelerator is not None:
            return self.accelerator.device
        else:
            return torch.device("cpu")
    
    def _move_to_device(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return inputs

    @torch.no_grad()
    def compute_nlls(self, dataloader):
        self.model.eval()

        all_query_ids = []
        all_nlls = []
        for step, inputs in enumerate(tqdm(dataloader, desc='Computing NLLs')):
            # move to gpu
            inputs = self._move_to_device(inputs)
            
            return_query_id = False
            if 'query_id' in inputs:
                query_id = inputs.pop("query_id") # batch_size
                return_query_id = True

            outputs = self.model(**inputs)            

            if self.model.config.is_encoder_decoder:
                shifted_logits = outputs.logits
                shifted_labels = inputs["labels"]
            else:
                shifted_logits = outputs.logits[:, :-1].contiguous()      # batch_size, seq_len - 1, vocab_size
                shifted_labels = inputs["labels"][:, 1:].contiguous()   # batch_size, seq_len - 1, vocab_size
            batch_size = shifted_logits.shape[0]

            token_loss = torch.nn.functional.cross_entropy(shifted_logits.flatten(0, 1), shifted_labels.view(-1), reduction="none").reshape(batch_size, -1)   # batch_size, seq_len - 1
            batch_loss = token_loss.sum(-1) # batch_size
            valid_token_num = (inputs["labels"] != -100).sum(-1)  # batch_size
            nll = batch_loss / valid_token_num   # batch_size

            if self.accelerator is not None:
                if return_query_id:
                    query_id = self.accelerator.gather_for_metrics(query_id)
                nll = self.accelerator.gather_for_metrics(nll)

            all_nlls.extend(nll.tolist())
            if return_query_id:
                all_query_ids.extend(query_id.tolist())
            
            # print(outputs.loss)
            # print(self.tokenizer.batch_decode(inputs["input_ids"]))
            # labels = inputs["labels"]
            # labels[labels == -100] = 0
            # print(self.tokenizer.batch_decode(labels))
            # print(all_nlls)
            # input()
                
        if return_query_id:
            return all_query_ids, all_nlls
        return all_nlls
    

    @torch.no_grad()
    def generate(self, dataloader, return_new_tokens_only=True, decode=True, **gen_kwargs):
        self.model.eval()
        
        all_query_ids = []
        all_generations = []
        
        for step, inputs in enumerate(tqdm(dataloader, desc='Generating')):
            # move to gpu
            inputs = self._move_to_device(inputs)
            
            return_query_id = False
            if 'query_id' in inputs:
                query_id = inputs.pop("query_id") # batch_size
                return_query_id = True

            outputs = self.model.generate(**inputs, **gen_kwargs)

            if return_new_tokens_only:
                if self.model.config.is_encoder_decoder:
                    if "decoder_input_ids" in inputs:
                        start_idx = inputs["decoder_input_ids"].shape[1] + 1
                    else:
                        start_idx = 1
                else:
                    start_idx = inputs["input_ids"].shape[1]
                outputs = outputs[:, start_idx:]

            if self.accelerator is not None:
                if return_query_id:
                    query_id = self.accelerator.gather_for_metrics(query_id)
                # must be contiguous
                outputs = outputs.contiguous()
                # FIXME: dim cannot be -1
                outputs = self.accelerator.pad_across_processes(outputs, pad_index=self.tokenizer.pad_token_id, dim=1)
                outputs = self.accelerator.gather_for_metrics(outputs)
                
            outputs = outputs.tolist()
            if decode:
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            all_generations.extend(outputs)
            
            if return_query_id:
                query_id = query_id.tolist()
                all_query_ids.extend(query_id)

        if return_query_id:
            return all_query_ids, all_generations
        return all_generations

