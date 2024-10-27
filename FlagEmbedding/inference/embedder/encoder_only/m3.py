import math
import torch
import queue
import logging
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Queue
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Any, List, Union, Dict, Literal, Tuple, Optional

from FlagEmbedding.abc.inference import AbsEmbedder
from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
    EncoderOnlyEmbedderM3ModelForInference, EncoderOnlyEmbedderM3Runner
)

logger = logging.getLogger(__name__)


class M3Embedder(AbsEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: str = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        devices: Union[str, List[str]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for M3Embedder
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        cache_dir: str = None,
        colbert_dim: int = -1,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        instruction: str = None,
        instruction_format: str = "{}{}",
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
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
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
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
            tokenizer=self.tokenizer,
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

    def compute_lexical_matching_score(
        self,
        lexical_weights_1: Union[Dict[str, float], List[Dict[str, float]]],
        lexical_weights_2: Union[Dict[str, float], List[Dict[str, float]]]
    ) -> Union[np.ndarray, float]:
        def _compute_single_lexical_matching_score(lw1: Dict[str, float], lw2: Dict[str, float]):
            scores = 0
            for token, weight in lw1.items():
                if token in lw2:
                    scores += weight * lw2[token]
            return scores

        if isinstance(lexical_weights_1, dict) and isinstance(lexical_weights_2, dict):
            return _compute_single_lexical_matching_score(lexical_weights_1, lexical_weights_2)
        elif isinstance(lexical_weights_1, list) and isinstance(lexical_weights_2, list):
            scores_array = []
            for lw1 in lexical_weights_1:
                scores_array.append([
                    _compute_single_lexical_matching_score(lw1, lw2)
                    for lw2 in lexical_weights_2
                ])
            return np.array(scores_array)
        else:
            raise ValueError("The input format of lexical_weights is not correct.")

    def colbert_score(self, q_reps, p_reps):
        q_reps, p_reps = torch.from_numpy(q_reps), torch.from_numpy(p_reps)
        token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / q_reps.size(0)
        return scores

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        return_dense: Optional[bool] = None,
        return_sparse: Optional[bool] = None,
        return_colbert_vecs: Optional[bool] = None,
        **kwargs: Any
    ) -> Dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
        Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]]
    ]:
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.query_max_length
        if return_dense is None: return_dense = self.return_dense
        if return_sparse is None: return_sparse = self.return_sparse
        if return_colbert_vecs is None: return_colbert_vecs = self.return_colbert_vecs

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
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        return_dense: Optional[bool] = None,
        return_sparse: Optional[bool] = None,
        return_colbert_vecs: Optional[bool] = None,
        **kwargs: Any
    ) -> Dict[
        Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
        Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]]
    ]:
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if return_dense is None: return_dense = self.return_dense
        if return_sparse is None: return_sparse = self.return_sparse
        if return_colbert_vecs is None: return_colbert_vecs = self.return_colbert_vecs

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
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        return_dense: Optional[bool] = None,
        return_sparse: Optional[bool] = None,
        return_colbert_vecs: Optional[bool] = None,
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

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

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
            return_tensors='pt',
            **kwargs
        ).to(device)
        while flag is False:
            try:
                test_inputs_batch = {}
                for k, v in max_length_inputs.items():
                    test_inputs_batch[k] = v.repeat(batch_size, 1)
                outputs = self.model(
                    test_inputs_batch,
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

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        max_query_length: Optional[int] = None,
        max_passage_length: Optional[int] = None,
        weights_for_different_modes: List[float] = None,
        **kwargs: Any
    ) -> Dict[
        Literal["colbert", "sparse", "dense", "sparse+dense", "colbert+sparse+dense"],
        List[float]
    ]:
        if batch_size is None: batch_size = self.batch_size
        if max_query_length is None: max_query_length = self.query_max_length
        if max_passage_length is None: max_passage_length = self.passage_max_length

        if len(self.target_devices) == 1:
            return self.compute_score_single_device(
                sentence_pairs,
                batch_size=batch_size,
                max_query_length=max_query_length,
                max_passage_length=max_passage_length,
                weights_for_different_modes=weights_for_different_modes,
                device=self.target_devices[0],
                **kwargs
            )

        pool = self.start_multi_process_pool(M3Embedder._compute_score_multi_process_worker)
        embeddings = self.compute_score_multi_process(
            sentence_pairs,
            pool,
            batch_size=batch_size,
            max_query_length=max_query_length,
            max_passage_length=max_passage_length,
            weights_for_different_modes=weights_for_different_modes,
            **kwargs
        )
        self.stop_multi_process_pool(pool)
        return embeddings

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L877
    def compute_score_multi_process(
        self,
        sentence_pairs: List[Tuple[str, str]],
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs
    ):
        chunk_size = math.ceil(len(sentence_pairs) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence_pair in sentence_pairs:
            chunk.append(sentence_pair)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, chunk, kwargs]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks")],
            key=lambda x: x[0],
        )

        scores_dict = self._concatenate_compute_score_results_from_multi_process([result[1] for result in results_list])
        return scores_dict

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L976
    @staticmethod
    def _compute_score_multi_process_worker(
        target_device: str, model: 'M3Embedder', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.compute_score_single_device(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    @torch.no_grad()
    def compute_score_single_device(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        max_query_length: int = 512,
        max_passage_length: int = 512,
        weights_for_different_modes: List[float] = None,
        device: str = None,
        **kwargs: Any
    ) -> Dict[
        Literal["colbert", "sparse", "dense", "sparse+dense", "colbert+sparse+dense"],
        List[float]
    ]:
        def _tokenize(texts: list, max_length: int):
            return self.tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                return_tensors='pt',
                **kwargs
            )

        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        if isinstance(sentence_pairs, list) and len(sentence_pairs) == 0:
            return []
        if isinstance(sentence_pairs[0], str):
            one_input_pair = True
            sentence_pairs = [sentence_pairs]
        else:
            one_input_pair = False

        all_scores = {
            'colbert': [],
            'sparse': [],
            'dense': [],
            'sparse+dense': [],
            'colbert+sparse+dense': []
        }
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]

            queries_batch = [pair[0] for pair in sentences_batch]
            corpus_batch = [pair[1] for pair in sentences_batch]

            queries_inputs = _tokenize(queries_batch, max_length=max_query_length).to(device)
            corpus_inputs = _tokenize(corpus_batch, max_length=max_passage_length).to(device)

            queries_output = self.model(
                queries_inputs,
                return_dense=True, return_sparse=True, return_colbert_vecs=True,
                return_sparse_embedding=True
            )
            corpus_output = self.model(
                corpus_inputs,
                return_dense=True, return_sparse=True, return_colbert_vecs=True,
                return_sparse_embedding=True
            )

            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = queries_output['dense_vecs'], queries_output['sparse_vecs'], \
            queries_output['colbert_vecs']
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = corpus_output['dense_vecs'], corpus_output['sparse_vecs'], \
            corpus_output['colbert_vecs']

            dense_scores = self.model.compute_dense_score(q_dense_vecs, p_dense_vecs)
            sparse_scores = self.model.compute_sparse_score(q_sparse_vecs, p_sparse_vecs)
            colbert_scores = self.model.compute_colbert_score(
                q_colbert_vecs, p_colbert_vecs,
                q_mask=queries_inputs['attention_mask']
            )

            if weights_for_different_modes is None:
                weights_for_different_modes = [1., 1., 1.]
                weight_sum = 3
                logger.info("default weights for dense, sparse, colbert are [1.0, 1.0, 1.0] ")
            else:
                assert len(weights_for_different_modes) == 3
                weight_sum = sum(weights_for_different_modes)

            inx = torch.arange(0, len(sentences_batch))
            dense_scores, sparse_scores, colbert_scores = dense_scores[inx, inx].float(), sparse_scores[
                inx, inx].float(), colbert_scores[inx, inx].float()

            all_scores['colbert'].extend(
                colbert_scores.cpu().numpy().tolist()
            )
            all_scores['sparse'].extend(
                sparse_scores.cpu().numpy().tolist()
            )
            all_scores['dense'].extend(
                dense_scores.cpu().numpy().tolist()
            )
            all_scores['sparse+dense'].extend(
                ((sparse_scores * weights_for_different_modes[1] + dense_scores * weights_for_different_modes[0])/(weights_for_different_modes[1]+weights_for_different_modes[0])).cpu().numpy().tolist()
            )
            all_scores['colbert+sparse+dense'].extend(
                ((colbert_scores * weights_for_different_modes[2] + sparse_scores * weights_for_different_modes[1] + dense_scores * weights_for_different_modes[0])/weight_sum).cpu().numpy().tolist()
            )

        if one_input_pair:
            return {k: v[0] for k, v in all_scores.items()}
        return all_scores

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

    def _concatenate_compute_score_results_from_multi_process(
        self,
        results_list: List[Dict[Literal["colbert", "sparse", "dense", "sparse+dense", "colbert+sparse+dense"], List[float]]]
    ):
        merged_results = {
            "colbert": [],
            "sparse": [],
            "dense": [],
            "sparse+dense": [],
            "colbert+sparse+dense": []
        }
        for key in merged_results.keys():
            for results in results_list:
                merged_results[key].extend(results[key])

        return merged_results