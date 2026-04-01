from typing import cast, Any, List, Union, Optional

import torch
import numpy as np

from .base import BaseLLMEmbedder, last_token_pool


class PseudoMoELLMEmbedder(BaseLLMEmbedder):
    """Decoder-only embedder for pseudo MoE checkpoints.

    This class follows the same behavior as :class:`BaseLLMEmbedder`, but supports
    selecting an active domain (e.g. ``general``, ``coding``, ``reasoning``) during
    inference when the underlying model implements domain routing.
    """

    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        use_bf16: bool = True,
        query_instruction_for_retrieval: Optional[str] = None,
        query_instruction_format: str = "Instruct: {}\nQuery: {}",
        devices: Optional[Union[str, List[str]]] = None,
        trust_remote_code: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        truncate_dim: Optional[int] = None,
        domain_for_pseudo_moe: Optional[str] = None,
        **kwargs: Any,
    ):
        self.domain_for_pseudo_moe = domain_for_pseudo_moe
        super().__init__(
            model_name_or_path=model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            truncate_dim=truncate_dim,
            **kwargs,
        )

    def _resolve_domain(self, kwargs: Any) -> Optional[str]:
        domain = kwargs.pop("domain_for_pseudo_moe", None)
        if domain is None:
            domain = kwargs.pop("domain", None)
        if domain is None:
            domain = self.domain_for_pseudo_moe
        return domain

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

        if device == "cpu":
            self.model.float()

        self.model.to(device)
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        domain = self._resolve_domain(kwargs)
        if domain is not None and hasattr(self.model, "set_domain"):
            self.model.set_domain(domain)

        model_forward_kwargs = {"return_dict": True}
        if domain is not None:
            model_forward_kwargs["domain"] = domain

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
        while flag is False:
            try:
                inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[: batch_size],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                try:
                    last_hidden_state = self.model(**inputs_batch, **model_forward_kwargs).last_hidden_state
                except TypeError:
                    last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
                _ = last_token_pool(last_hidden_state, inputs_batch['attention_mask'])
                flag = True
            except RuntimeError:
                batch_size = batch_size * 3 // 4
            except torch.cuda.OutOfMemoryError:
                batch_size = batch_size * 3 // 4

        # encode
        all_embeddings = []
        for start_index in range(0, len(sentences), batch_size):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)
            try:
                last_hidden_state = self.model(**inputs_batch, **model_forward_kwargs).last_hidden_state
            except TypeError:
                last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
            embeddings = last_token_pool(last_hidden_state, inputs_batch['attention_mask'])
            embeddings = self._truncate_embeddings(embeddings)
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e4, neginf=-1e4)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = self._convert_to_numpy(embeddings, device=device)
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        # adjust the order of embeddings
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings
