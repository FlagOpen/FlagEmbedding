import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Any, List, Union, Dict, Literal

from FlagEmbedding.abc.inference import AbsEmbedder
from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
    EncoderOnlyEmbedderM3ModelForInference, EncoderOnlyEmbedderM3Runner
)


class M3Embedder(AbsEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = False,
        use_fp16: bool = False,
        query_instruction_for_retrieval: str = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        devices: Union[str, List[str]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for M3Embedder
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: str = None,
        colbert_dim: int = -1,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            **kwargs
        )
        self.pooling_method = pooling_method
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.model = EncoderOnlyEmbedderM3ModelForInference(
            EncoderOnlyEmbedderM3Runner.get_model(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                colbert_dim=colbert_dim,
                cache_dir=cache_dir
            ),
            sentence_pooling_method=pooling_method,
            normalize_embeddings=normalize_embeddings
        )
    
    def convert_id_to_token(self, lexical_weights: List[Dict]):
        if isinstance(lexical_weights, dict):
            lexical_weights = [lexical_weights]
        new_lexical_weights = []
        for item in lexical_weights:
            new_item = {}
            for id, weight in item.items():
                token = self.tokenizer.decode([int(id)])
                new_item[token] = weight
            new_lexical_weights.append(new_item)

        if len(new_lexical_weights) == 1:
            new_lexical_weights = new_lexical_weights[0]
        return new_lexical_weights

    def compute_lexical_matching_score(self, lexical_weights_1: Dict[str, float], lexical_weights_2: Dict[str, float]):
        scores = 0
        for token, weight in lexical_weights_1.items():
            if token in lexical_weights_2:
                scores += weight * lexical_weights_2[token]
        return scores

    def colbert_score(self, q_reps, p_reps):
        q_reps, p_reps = torch.from_numpy(q_reps), torch.from_numpy(p_reps)
        token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / q_reps.size(0)
        return scores
    
    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        **kwargs: Any
    ) -> Dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
        Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]]
    ]:
        return super().encode_queries(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
            **kwargs
        )
    
    def encode_corpus(
        self,
        queries: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        **kwargs: Any
    ) -> Dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
        Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]]
    ]:
        return super().encode_corpus(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
            **kwargs
        )
    
    def encode(
        self,
        queries: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        **kwargs: Any
    ) -> Dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
        Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]]
    ]:
        return super().encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
            **kwargs
        )
    
    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        device: str = None,
        **kwargs: Any
    ):        
        if device is None:
            device = self.target_devices[0]
        
        self.model.to(device)
        self.model.eval()
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True
        
        def _process_token_weights(token_weights: np.ndarray, input_ids: list):
            # conver to dict
            result = defaultdict(int)
            unused_tokens = set()
            for _token in ['cls_token', 'eos_token', 'pad_token', 'unk_token']:
                if _token in self.tokenizer.special_tokens_map:
                    _token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map[_token])
                    unused_tokens.add(_token_id)
            # token_weights = np.ceil(token_weights * 100)
            for w, idx in zip(token_weights, input_ids):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    # w = int(w)
                    if w > result[idx]:
                        result[idx] = w
            return result

        def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
            # delte the vectors of padding tokens
            tokens_num = np.sum(attention_mask)
            return colbert_vecs[:tokens_num - 1]  # we don't use the embedding of cls, so select tokens_num-1
        
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
            max_length=max_length,
            return_tensors='pt',
            **kwargs
        ).to(device)
        while flag is False:
            try:
                test_inputs_batch = {}
                for k, v in max_length_inputs.items():
                    test_inputs_batch[k] = v.repeat(batch_size, 1)
                outputs = self.model(
                    inputs_batch,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert_vecs
                )
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
        
        # encode
        all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                max_length=max_length,
                return_tensors='pt',
                **kwargs
            ).to(device)
            outputs = self.model(
                inputs_batch,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert_vecs
            )
            
            if return_dense:
                all_dense_embeddings.append(outputs['dense_vecs'].cpu().numpy())

            if return_sparse:
                token_weights = outputs['sparse_vecs'].squeeze(-1)
                all_lexical_weights.extend(
                    list(map(
                        _process_token_weights, 
                        token_weights.cpu().numpy(),
                        inputs_batch['input_ids'].cpu().numpy().tolist()
                )))

            if return_colbert_vecs:
                all_colbert_vecs.extend(
                    list(map(
                        _process_colbert_vecs,
                        outputs['colbert_vecs'].cpu().numpy(),
                        inputs_batch['attention_mask'].cpu().numpy()
                )))

        if return_dense:
            all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
            # adjust the order of embeddings
            all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]
            if input_was_string:
                all_dense_embeddings = all_dense_embeddings[0]
        else:
            all_dense_embeddings = None

        if return_sparse:
            # adjust the order of lexical weights
            all_lexical_weights = [all_lexical_weights[i] for i in np.argsort(length_sorted_idx)]
            if input_was_string:
                all_lexical_weights = all_lexical_weights[0]
        else:
            all_lexical_weights = None

        if return_colbert_vecs:
            # adjust the order of embeddings
            all_colbert_vecs = [all_colbert_vecs[i] for i in np.argsort(length_sorted_idx)]
            if input_was_string:
                all_colbert_vecs = all_colbert_vecs[0]
        else:
            all_colbert_vecs = None
        
        # return the embeddings
        return {
            "dense_vecs": all_dense_embeddings,
            "lexical_weights": all_lexical_weights,
            "colbert_vecs": all_colbert_vecs
        }

    def _concatenate_results_from_multi_process(
        self,
        results_list: List[Dict[Literal["dense_vecs", "lexical_weights", "colbert_vecs"], Any]]
    ):
        merged_results = {
            "dense_vecs": [],
            "lexical_weights": [],
            "colbert_vecs": []
        }
        for key in merged_results.keys():
            for results in results_list:
                if results[key] is None:
                    merged_results[key] = None
                    break
                else:
                    if key == "dense_vecs":
                        merged_results[key].append(results[key])
                    else:
                        merged_results[key].extend(results[key])
        
        if merged_results["dense_vecs"] is not None:
            merged_results["dense_vecs"] = np.concatenate(merged_results["dense_vecs"], axis=0)
        
        return merged_results
