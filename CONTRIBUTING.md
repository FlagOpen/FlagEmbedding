Hey, I recently used this library to use the BAAI/bge-en-icl for embedding text data in OCEAN form. It kept crashing out when I tried to run it on my local GPU using the FlagEmbedding and it kept returning "killed"

I made a small add on with the accelerate library which enabled it to split the model into CPU and GPU when it overflows. Within this file FlagEmbedding/FlagEmbedding/flag_models.py make the following changes Just replacing the following text block fixed those issues!!


```python
from typing import cast, List, Union
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
import torch
from torch import Tensor
import torch.nn.functional as F


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'


def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'


class FlagICLModel:
    def __init__(
        self,
        model_name_or_path: str = None,
        normalize_embeddings: bool = True,
        query_instruction_for_retrieval: str = 'Given a query, retrieval relevant passages that answer the query.',
        examples_for_task: List[dict] = None,
        use_fp16: bool = True
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Use device_map="auto" to offload to CPU when necessary
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            device_map="auto",  # Automatically offload model components between GPU and CPU
            offload_folder="offload"  # Directory to temporarily store offloaded weights (useful if offloading is needed)
        )

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.examples_for_task = examples_for_task

        self.set_examples()
        self.suffix = '\n<response>'

        self.normalize_embeddings = normalize_embeddings

        # Determine device for setting up other components, but no need to manually move the model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False

        # Do not move the model explicitly if it is already being managed by Accelerate
        if use_fp16 and isinstance(self.model, torch.nn.Module) and not hasattr(self.model, "hf_device_map"):
            self.model.half()

        # Handle multiple GPUs if available
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus} GPUs----------")
            self.model = torch.nn.DataParallel(self.model)
```
