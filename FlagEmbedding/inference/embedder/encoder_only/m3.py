import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Any, List, Union, Dict
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available

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
        pooling_method: str = "cls",
        trust_remote_code: bool = False,
        query_instruction_for_retrieval: str = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        cache_dir: str = None,
        device: str = None, # specify device, such as "cuda:0"
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings,
            use_fp16
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
                colbert_dim=kwargs.get("colbert_dim", -1),
                cache_dir=cache_dir
            ),
            sentence_pooling_method=pooling_method,
            normalize_embeddings=normalize_embeddings
        )
        
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.kwargs = kwargs
        
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
        
        self.num_gpus = 1
        # if self.num_gpus > 1:
        #     print(f"----------using {self.num_gpus}*GPUs----------")
        #     self.model.model = torch.nn.DataParallel(self.model.model)
    
    @staticmethod
    def get_detailed_instruct(instruction_format: str, instruction: str, query: str):
        return instruction_format.format(instruction, query)
    
    def encode_queries_single_gpu(
        self,
        queries: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        **kwargs: Any
    ):
        if self.query_instruction_for_retrieval is not None:
            if isinstance(queries, str):
                input_texts = self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_retrieval, queries)
            else:
                input_texts = [self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_retrieval, query) for query in queries]
        else:
            input_texts = queries
        return self.encode(
            input_texts,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
    
    def encode_corpus_single_gpu(
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

    def compute_lexical_matching_score(self, lexical_weights_1: Dict, lexical_weights_2: Dict):
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
    
    @torch.no_grad()
    def encode(
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
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus
        self.model.eval()

        if device is None:
            device = self.device

        self.model.to(device)
        
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
