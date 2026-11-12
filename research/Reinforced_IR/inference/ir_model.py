import os
from typing import List, Union, Optional

from FlagEmbedding.inference.embedder.model_mapping import (
    EmbedderModelClass,
    AUTO_EMBEDDER_MAPPING, EMBEDDER_CLASS_MAPPING
)

from FlagEmbedding import FlagAutoModel

from agent import GPTAgent, LLMAgent, LLMInstructAgent

prompt_template = """\
Given a retrieval task and a query, your mission is to generate a brief {answer_type} for the query in the context of the retrieval task.
Please generate without any explanation.

Task: {task}

Query: {query}

Your output:"""

class Reinforced_IR_Model():
    def __init__(
        self,
        model_name_or_path: str,
        model_class: Optional[Union[str, EmbedderModelClass]] = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        devices: Optional[Union[str, List[str]]] = None,
        pooling_method: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        query_instruction_format: Optional[str] = None,
        generator_model_name_or_path: Optional[str] = None,
        temperature: float = 1.0,
        gpu_memory_utilization: float = 0.5,
        tensor_parallel_size: int = None,
        top_p: float = 1.0,
        max_tokens: int = 300,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_type: str = "llm_instruct",
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_class = model_class
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.devices = devices
        self.pooling_method = pooling_method
        self.trust_remote_code = trust_remote_code
        self.query_instruction_format = query_instruction_format
        self.generator_model_name_or_path = generator_model_name_or_path
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.model_type = model_type
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

        self.generator = None
        self.retriever = None

    def load_retriever(self):
        if self.retriever is None:
            self.retriever = FlagAutoModel.from_finetuned(
                model_name_or_path=self.model_name_or_path,
                model_class=self.model_class,
                normalize_embeddings=self.normalize_embeddings,
                use_fp16=self.use_fp16,
                query_instruction_for_retrieval=self.query_instruction_for_retrieval,
                devices=self.devices,
                pooling_method=self.pooling_method,
                trust_remote_code=self.trust_remote_code,
                query_instruction_format=self.query_instruction_format,
                **self.kwargs,
            )
        self.offload_generator()

    def load_generator(self):
        if self.generator_model_name_or_path is not None:
            self.offload_retriever()
        if self.generator is None and self.generator_model_name_or_path is not None:
            if self.model_type == 'llm':
                self.generator = LLMAgent(model_name=self.generator_model_name_or_path,
                                          gpu_memory_utilization=self.gpu_memory_utilization,
                                          tensor_parallel_size=self.tensor_parallel_size)
            elif self.model_type == 'llm_instruct':
                self.generator = LLMInstructAgent(generate_model_path=self.generator_model_name_or_path,
                                                  gpu_memory_utilization=self.gpu_memory_utilization,
                                                  tensor_parallel_size=self.tensor_parallel_size)
            else:
                self.generator = GPTAgent(model_name=self.generator_model_name_or_path,
                                          api_key=self.api_key,
                                          base_url=self.base_url)

    def offload_retriever(self):
        if self.retriever is not None:
            del self.retriever
            self.retriever = None

    def offload_generator(self):
        if self.generator is not None:
            del self.generator
            self.generator = None

    def encode_queries(self, task_instruction, answer_type, queries, **kwargs):
        prompts = [prompt_template.format(
            answer_type=answer_type,
            task=task_instruction,
            query=query
        ) for query in queries]
        self.load_generator()
        if self.generator is not None:
            augmented_queries = self.generator.generate(prompts, **kwargs)
            print(augmented_queries)
            augmented_queries = ['Generate the topic about this passage: ' + e for e in augmented_queries]
        self.load_retriever()
        if self.generator is not None:
            return self.retriever.encode_corpus(augmented_queries, **kwargs) * 0.2 + \
                self.retriever.encode_queries(queries, **kwargs) * 0.8
        return self.retriever.encode_queries(queries, **kwargs)

    def encode_corpus(self, corpus, **kwargs):
        self.load_retriever()
        return self.retriever.encode_corpus(corpus, **kwargs)

    def encode(self, corpus, **kwargs):
        self.load_retriever()
        return self.retriever.encode(corpus, **kwargs)