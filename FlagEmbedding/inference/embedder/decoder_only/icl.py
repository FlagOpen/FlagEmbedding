from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Optional

import queue
from multiprocessing import Queue

import gc
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from FlagEmbedding.abc.inference import AbsEmbedder


# Pooling function for LLM-based embedding models
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """Last token pooling method.

    Args:
        last_hidden_state (torch.Tensor): The last hidden state of the model.
        attention_mask (torch.Tensor): Attention mask. Defaults to :data:`None`.

    Returns:
        torch.Tensor: The embedding vectors after pooling.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class ICLLLMEmbedder(AbsEmbedder):
    """
    Embedder class for BGE-EN-icl.
    
    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to :data:`True`.
        use_fp16 (bool, optional) If true, use half-precision floating-point to speed up computation with a slight performance 
            degradation. Defaults to :data:`True`.
        query_instruction_for_retrieval (Optional[str], optional): Query instruction for retrieval tasks, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`None`.
        query_instruction_format (str, optional): The template for :attr:`query_instruction_for_retrieval`. Defaults to :data:`"{}{}"`.
        devices (Optional[Union[str, int, List[str], List[int]]], optional): Devices to use for model inference. Defaults to :data:`None`.
        examples_for_task (Optional[List[dict]], optional): Few-shot examples for the model to enhance model's ability. 
            Defaults to :data:`None`.
        examples_instruction_format (str, optional): Example format when using :attr:`examples_for_task`.
        trust_remote_code (bool, optional): trust_remote_code for HF datasets or models. Defaults to :data:`False`.
        cache_dir (Optional[str], optional): Cache directory for the model. Defaults to :data:`None`.
        batch_size (int, optional): Batch size for inference. Defaults to :data:`256`.
        query_max_length (int, optional): Maximum length for query. Defaults to :data:`512`.
        passage_max_length (int, optional): Maximum length for passage. Defaults to :data:`512`.
        convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will be a Torch Tensor. 
            Defaults to :data:`True`.
    
    Attributes:
        DEFAULT_POOLING_METHOD: The default pooling method when running the model.
    """
    DEFAULT_POOLING_METHOD = "last_token"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "<instruct>{}\n<query>{}", # specify the format of query_instruction_for_retrieval
        suffix: str = '\n<response>',
        devices: Optional[Union[str, List[str]]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for ICLLLMEmbedder
        examples_for_task: Optional[List[dict]] = None,
        examples_instruction_format: str = "<instruct>{}\n<query>{}\n<response>{}", # specify the format of examples_for_task
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ):
        query_instruction_format = query_instruction_format.replace('\\n', '\n')
        examples_instruction_format = examples_instruction_format.replace('\\n', '\n')
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.examples_for_task = examples_for_task
        self.examples_instruction_format = examples_instruction_format

        if self.kwargs.get("pooling_method", "last_token") != "last_token":
            raise ValueError("Pooling method must be 'last_token' for LLM-based models.")

        self.set_examples()
        self.suffix = suffix

        self.query_pool = None
    
    def __del__(self):
        self.stop_self_pool()
        self.stop_self_query_pool()

    def set_examples(self, examples_for_task: Optional[List[dict]] = None):
        """Set the prefix to the provided examples.

        Args:
            examples_for_task (Optional[List[dict]], optional): Few-shot examples for the model to enhance model's ability. 
                Defaults to :data:`None`.
        """
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
    def get_detailed_example(instruction_format: str, instruction: str, query: str, response: str):
        """Combine the instruction and sentence along with the instruction format.

        Args:
            instruction_format (str): Format for instruction.
            instruction (str): The text of instruction.
            query (str): The text of example query.
            response (str): The text of example response.

        Returns:
            str: The complete example following the given format.
        """
        return instruction_format.format(instruction, query, response)

    def stop_self_query_pool(self):
        if self.query_pool is not None:
            self.stop_multi_process_pool(self.query_pool)
            self.query_pool = None
        try:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        except:
            pass
        gc.collect()

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode the queries.

        Args:
            queries (Union[List[str], str]): Input queries to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: Return the embedding vectors in a numpy array or tensor.
        """
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.query_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy
        
        if isinstance(queries, str) or len(self.target_devices) == 1:
            return self.encode_queries_single_device(
                queries,
                batch_size=batch_size,
                max_length=max_length,
                convert_to_numpy=convert_to_numpy,
                device=self.target_devices[0],
                **kwargs
            )

        self.stop_self_pool()
        if self.query_pool is None:
            self.query_pool = self.start_multi_process_pool(ICLLLMEmbedder._encode_queries_multi_process_worker)
        embeddings = self.encode_multi_process(
            queries,
            self.query_pool,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        return embeddings

    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode the corpus.

        Args:
            corpus (Union[List[str], str]): Input corpus to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: Return the embedding vectors in a numpy array or tensor.
        """
        self.stop_self_query_pool()
        return super().encode_corpus(
            corpus,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        """Encode the input sentences with the embedding model.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (Optional[int], optional): Number of sentences for each iter. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            convert_to_numpy (Optional[bool], optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`None`.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        return super().encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L976
    @staticmethod
    def _encode_queries_multi_process_worker(
        target_device: str, model: 'ICLLLMEmbedder', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.encode_queries_single_device(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    @torch.no_grad()
    def encode_queries_single_device(
        self,
        queries: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any
    ):
        """Encode queries by a single device.

        Args:
            queries (Union[List[str], str]): Input queries to encode.
            batch_size (int, optional): Number of queries for each iter. Defaults to :data:`256`.
            max_length (int, optional): Maximum length of tokens. Defaults to :data:`512`.
            convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`True`.
            device (Optional[str], optional): Device to use for encoding. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        input_was_string = False
        if isinstance(queries, str):
            queries = [queries]
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
        _len_2 = len(self.tokenizer(f'{self.suffix}</s>', add_special_tokens=False)['input_ids'])
        new_max_length = (len(prefix_ids) + len(suffix_ids) + max_length + 8) // 8 * 8 + 8

        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in trange(0, len(input_texts), batch_size, desc='pre tokenize'):
            sentences_batch = input_texts[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length - _len_1 - _len_2,
                add_special_tokens=False,
                **kwargs
            )
            sentences_batch = self.tokenizer.batch_decode(inputs_batch['input_ids'])
            for i in range(len(sentences_batch)):
                sentences_batch[i] = self.prefix + sentences_batch[i] + self.suffix
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=new_max_length,
                **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        sentences_sorted = [input_texts[i] for i in length_sorted_idx]

        # adjust batch size
        flag = False
        while flag is False:
            try:
                inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[: batch_size],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
                embeddings = last_token_pool(last_hidden_state, inputs_batch['attention_mask'])
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
            except torch.OutofMemoryError as e:
                batch_size = batch_size * 3 // 4

        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences_sorted), batch_size), desc="Inference Embeddings",
                                disable=len(sentences_sorted) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)

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

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any
    ):
        """Encode input sentences by a single device.

        Args:
            sentences (Union[List[str], str]): Input sentences to encode.
            batch_size (int, optional): Number of sentences for each iter. Defaults to :data:`256`.
            max_length (int, optional): Maximum length of tokens. Defaults to :data:`512`.
            convert_to_numpy (bool, optional): If True, the output embedding will be a Numpy array. Otherwise, it will 
                be a Torch Tensor. Defaults to :data:`True`.
            device (Optional[str], optional): Device to use for encoding. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: return the embedding vectors in a numpy array or tensor.
        """
        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in trange(0, len(sentences), batch_size, desc='pre tokenize',
                                  disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # adjust batch size
        flag = False
        while flag is False:
            try:
                inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[: batch_size],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
                embeddings = last_token_pool(last_hidden_state, inputs_batch['attention_mask'])
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
            except torch.OutofMemoryError as e:
                batch_size = batch_size * 3 // 4

        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)
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
