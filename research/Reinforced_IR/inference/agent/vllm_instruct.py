import torch

from typing import List, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest


class LLMInstructAgent():
    def __init__(
        self,
        generate_model_path: str = None,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(generate_model_path, trust_remote_code=True)
        self.llm = LLM(model=generate_model_path, gpu_memory_utilization=gpu_memory_utilization,
                       trust_remote_code=True,
                       tensor_parallel_size=torch.cuda.device_count() if tensor_parallel_size is None else tensor_parallel_size,
                       enable_lora=True, max_lora_rank=64)
        self.model_name = generate_model_path.lower()

    def generate(
        self,
        prompts: Union[List[str], str] = None,
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 300,
        stop: List[str] = [],
        repetition_penalty: float = 1.1,
        lora_path: str = None,
        n: int = 1
    ):

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                         stop=stop, n=n,
                                         repetition_penalty=repetition_penalty,
                                         stop_token_ids=[128001, 128009] if 'llama' in self.model_name else None)
        lora_request = None
        if lora_path is not None:
            lora_request = LoRARequest("lora_adapter", 1, lora_path)

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [self.tokenizer.decode(
            self.tokenizer.apply_chat_template([
                {
                    "role": "user", "content": t[:5120]
                },
            ], add_generation_prompt=True, )[1:]) for t in prompts]
        outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_request)

        if n > 1:
            results = []
            for output in outputs:
                results.append([output.outputs[i].text.strip() for i in range(len(output.outputs))])
        return [output.outputs[0].text.strip() for output in outputs]

        # return [output.outputs[0].text.replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').replace('assistant<|end_header_id|>\n\n', '').replace('assistant<|end_header_id|>', '').strip() for output in outputs]

    def generate_direct(
        self,
        prompts: List = None,
        temperature: float = 0,
        top_p: float = 1.0,
        max_tokens: int = 300,
        stop: List[str] = [],
        repetition_penalty: float = 1.1,
        lora_path: str = None,
        n: int = 1
    ):

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                         stop=stop, n=n,
                                         repetition_penalty=repetition_penalty,
                                         stop_token_ids=[128001, 128009] if 'llama' in self.model_name else None)
        lora_request = None
        if lora_path is not None:
            lora_request = LoRARequest("lora_adapter", 1, lora_path)

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [self.tokenizer.decode(
            self.tokenizer.apply_chat_template(t, add_generation_prompt=True, )[1:]) for t in prompts]
        outputs = self.llm.generate(prompts, sampling_params, lora_request=lora_request)

        if n > 1:
            results = []
            for output in outputs:
                results.append([output.outputs[i].text.strip() for i in range(len(output.outputs))])
        return [output.outputs[0].text.strip() for output in outputs]