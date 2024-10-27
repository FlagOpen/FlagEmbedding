import torch
import numpy as np
from tqdm import tqdm, trange
from typing import Any, List, Union, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from FlagEmbedding.abc.inference import AbsReranker


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BaseReranker(AbsReranker):
    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        query_instruction_for_rerank: str = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_rerank
        passage_instruction_for_rerank: str = None,
        passage_instruction_format: str = "{}{}", # specify the format of passage_instruction_for_rerank
        trust_remote_code: bool = False,
        cache_dir: str = None,
        devices: Union[str, List[str], List[int]] = None, # specify devices, such as ["cuda:0"] or ["0"]
        # inference
        batch_size: int = 128,
        query_max_length: int = None,
        max_length: int = 512,
        normalize: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            query_instruction_for_rerank=query_instruction_for_rerank,
            query_instruction_format=query_instruction_format,
            passage_instruction_for_rerank=passage_instruction_for_rerank,
            passage_instruction_format=passage_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            max_length=max_length,
            normalize=normalize,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code, 
            cache_dir=cache_dir
        )

    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        device: str = None,
        **kwargs: Any
    ) -> List[float]:
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.max_length
        if query_max_length is None:
            if self.query_max_length is not None:
                query_max_length = self.query_max_length
            else:
                query_max_length = max_length * 3 // 4
        if normalize is None: normalize = self.normalize

        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize"):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            passages_inputs_batch = self.tokenizer(
                passages,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                **kwargs
            )['input_ids']
            for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
                item = self.tokenizer.prepare_for_model(
                    q_inp,
                    d_inp,
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                )
                all_inputs.append(item)
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # adjust batch size
        flag = False
        while flag is False:
            try:
                test_inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[:min(len(all_inputs_sorted), batch_size)],
                    padding=True,
                    return_tensors='pt',
                    **kwargs
                ).to(device)
                scores = self.model(**test_inputs_batch, return_dict=True).logits.view(-1, ).float()
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4

        all_scores = []
        for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Compute Scores",
                                disable=len(all_inputs_sorted) < 128):
            sentences_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer.pad(
                sentences_batch,
                padding=True,
                max_length=max_length,
                return_tensors='pt',
                **kwargs
            ).to(device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores