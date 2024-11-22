from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Optional

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


class BaseLLMEmbedder(AbsEmbedder):
    """Base embedder class for LLM like decoder only models.

    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        normalize_embeddings (bool, optional): If True, normalize the embedding vector. Defaults to :data:`True`.
        use_fp16 (bool, optional): If true, use half-precision floating-point to speed up computation with a slight performance 
            degradation. Defaults to :data:`True`.
        query_instruction_for_retrieval (Optional[str], optional): Query instruction for retrieval tasks, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`None`.
        query_instruction_format (str, optional): The template for :attr:`query_instruction_for_retrieval`. Defaults to :data:`"{}{}"`.
        devices (Optional[Union[str, int, List[str], List[int]]], optional): Devices to use for model inference. Defaults to :data:`None`.
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
        query_instruction_format: str = "Instruct: {}\nQuery: {}", # specify the format of query_instruction_for_retrieval
        devices: Optional[Union[str, List[str]]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for BaseLLMEmbedder
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ):
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

        if self.kwargs.get("pooling_method", "last_token") != "last_token":
            raise ValueError("Pooling method must be 'last_token' for LLM-based models.")

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
        return super().encode_queries(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

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

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: Optional[str] = None,
        **kwargs: Any   # add `pad_to_multiple_of=8` for bge-multilingual-gemmma2
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
