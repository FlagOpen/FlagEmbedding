# bge-en-icl
import torch
import numpy as np
from tqdm import tqdm
from typing import cast, Any, List, Union
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available

from src.abc.inference import AbsEmbedder


# Pooling function for LLM-based embedding models
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class ICLLLMEmbedder(AbsEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = False,
        use_fp16: bool = False,
        trust_remote_code: bool = False,
        query_instruction_for_retrieval: str = None,
        query_instruction_format: str = "<instruct>{}\n<query>{}", # specify the format of query_instruction_for_retrieval
        examples_for_task: List[dict] = None,
        examples_instruction_format: str = "<instruct>{}\n<query>{}\n<response>{}", # specify the format of examples_for_task
        device: str = None, # specify device, such as "cuda:0"
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings,
            use_fp16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.examples_for_task = examples_for_task
        self.examples_instruction_format = examples_instruction_format
        self.kwargs = kwargs
        
        self.set_examples()
        self.suffix = '\n<response>'
        
        if device is not None:
            self.device = torch.device(device)
            self.num_gpus = 1
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.num_gpus = torch.cuda.device_count()
            else:
                self.num_gpus = -1  # TODO: DataParallel for other devices
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                elif is_torch_npu_available():
                    self.device = torch.device("npu")
                else:
                    self.device = torch.device("cpu")
        
        if self.device.type == "cpu":
            self.use_fp16 = False
        
        if self.use_fp16: self.model.half()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)
    
    def set_examples(self, examples_for_task: List[dict] = None):
        if examples_for_task is None and self.examples_for_task is None:
            self.prefix = ''
        elif examples_for_task is not None:
            eg_paris = []
            for i in range(len(examples_for_task)):
                eg_paris.append(
                    self.get_detailed_example(
                        self.examples_instruction_format,
                        examples_for_task[i].get('instruct', self.query_instruction_for_retrieval),
                        examples_for_task[i].get('query', ''),
                        examples_for_task[i].get('response', '')
                    )
                )
            self.prefix = '\n\n'.join(eg_paris) + '\n\n'
        else:
            eg_paris = []
            for i in range(len(self.examples_for_task)):
                eg_paris.append(
                    self.get_detailed_example(
                        self.examples_instruction_format,
                        self.examples_for_task[i].get('instruct', self.query_instruction_for_retrieval),
                        self.examples_for_task[i].get('query', ''),
                        self.examples_for_task[i].get('response', '')
                    )
                )
            self.prefix = '\n\n'.join(eg_paris) + '\n\n'
    
    @staticmethod
    def get_detailed_instruct(instruction_format: str, instruction: str, query: str):
        return instruction_format.format(instruction, query)

    @staticmethod
    def get_detailed_example(instruction_format: str, instruction: str, query: str, response: str):
        return instruction_format.format(instruction, query, response)
    
    @torch.no_grad()
    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any
    ):
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_retrieval, queries)
            else:
                input_texts = [self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_retrieval, query) for query in queries]
        else:
            input_texts = queries
        
        prefix_ids = self.tokenizer(self.prefix, add_special_tokens=False)['input_ids']
        suffix_ids = self.tokenizer(self.suffix, add_special_tokens=False)['input_ids']
        
        _len_1 = len(self.tokenizer('<s>', add_special_tokens=False)['input_ids'])
        _len_2 = len(self.tokenizer('\n<response></s>', add_special_tokens=False)['input_ids'])
        
        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in range(0, len(input_texts), batch_size):
            sentences_batch = input_texts[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            all_inputs.extend(inputs_batch)
        
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        sentences_sorted = [input_texts[i] for i in length_sorted_idx]
        
        # adjust batch size
        flag = False
        max_length_inputs = self.tokenizer.pad(
            all_inputs_sorted[:1],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            **kwargs
        ).to(self.device)
        while flag is False:
            try:
                test_inputs_batch = {}
                for k, v in max_length_inputs.items():
                    test_inputs_batch[k] = v.repeat(batch_size, 1)
                last_hidden_state = self.model(**test_inputs_batch, return_dict=True).last_hidden_state
                embeddings = last_token_pool(last_hidden_state, test_inputs_batch['attention_mask'])
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
        
        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences_sorted), batch_size), desc="Inference Embeddings",
                                disable=len(sentences_sorted) < 256):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                max_length=max_length - _len_1 - _len_2,
                return_token_type_ids=False,
                truncation=True,
                return_tensors=None,
                add_special_tokens=False
            )
            new_max_length = (len(prefix_ids) + len(suffix_ids) + max_length + 8) // 8 * 8 + 8
            sentences_batch = self.tokenizer.batch_decode(inputs['input_ids'])
            for i in range(len(sentences_batch)):
                sentences_batch[i] = self.prefix + sentences_batch[i] + self.suffix
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=new_max_length,
                add_special_tokens=True
            ).to(self.device)

            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = last_token_pool(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
        
        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        # adjust the order of embeddings
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        
        # return the embeddings
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings
    
    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any
    ):
        passage_instruction_for_retrieval = self.kwargs.get("passage_instruction_for_retrieval", None)
        passage_instruction_format = self.kwargs.get("passage_instruction_format", "{}{}")
        if passage_instruction_for_retrieval is not None:
            if isinstance(corpus, str):
                input_texts = self.get_detailed_instruct(passage_instruction_format, passage_instruction_for_retrieval, corpus)
            else:
                input_texts = [self.get_detailed_instruct(passage_instruction_format, passage_instruction_for_retrieval, passage) for passage in corpus]
        else:
            input_texts = corpus
        return self.encode(
            input_texts,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
    
    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any   # add `pad_to_multiple_of=8` for bge-multilingual-gemmma2
    ):
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        
        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            all_inputs.extend(inputs_batch)
        
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        
        # adjust batch size
        flag = False
        max_length_inputs = self.tokenizer.pad(
            all_inputs_sorted[:1],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            **kwargs
        ).to(self.device)
        while flag is False:
            try:
                test_inputs_batch = {}
                for k, v in max_length_inputs.items():
                    test_inputs_batch[k] = v.repeat(batch_size, 1)
                last_hidden_state = self.model(**test_inputs_batch, return_dict=True).last_hidden_state
                embeddings = last_token_pool(last_hidden_state, test_inputs_batch['attention_mask'])
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
        
        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
                **kwargs
            ).to(self.device)
            last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
            embeddings = last_token_pool(last_hidden_state, inputs_batch['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        # adjust the order of embeddings
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        # return the embeddings
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings