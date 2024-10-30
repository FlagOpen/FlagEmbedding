from tqdm import tqdm
from typing import cast, Any, List, Union, Optional

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from FlagEmbedding.abc.inference import AbsEmbedder


class BaseEmbedder(AbsEmbedder):
    DEFAULT_POOLING_METHOD = "cls"

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        devices: Optional[Union[str, List[str]]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for BaseEmbedder
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        instruction: Optional[str] = None,
        instruction_format: str = "{}{}",
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
            instruction=instruction,
            instruction_format=instruction_format,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        self.pooling_method = pooling_method

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

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
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
        **kwargs: Any
    ):
        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

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
        for start_index in range(0, len(sentences), batch_size):
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
        max_length_inputs = self.tokenizer.pad(
            all_inputs_sorted[:1],
            padding=True,
            return_tensors='pt',
            **kwargs
        ).to(device)
        while flag is False:
            try:
                test_inputs_batch = {}
                for k, v in max_length_inputs.items():
                    test_inputs_batch[k] = v.repeat(batch_size, 1)
                last_hidden_state = self.model(**test_inputs_batch, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, test_inputs_batch['attention_mask'])
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
            embeddings = self.pooling(last_hidden_state, inputs_batch['attention_mask'])
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

    def pooling(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        else:
            raise NotImplementedError(f"pooling method {self.pooling_method} not implemented")
