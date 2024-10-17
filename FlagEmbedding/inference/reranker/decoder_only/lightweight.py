import torch
import warnings
import numpy as np
from tqdm import tqdm, trange
from typing import cast, Any, List, Union, Tuple
from peft import PeftModel
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_npu_available
from torch.utils.data import Dataset, DataLoader

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.inference.reranker.encoder_only.base import sigmoid

def last_logit_pool_lightweight(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)


class collater_for_lightweight():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8
        self.label_pad_token_id = -100
        warnings.filterwarnings("ignore",
                                message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy.")

    def __call__(self, data):
        features = data[0]
        query_lengths = data[1]
        prompt_lengths = data[2]

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        collected = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

        return collected, query_lengths, prompt_lengths


class LightweightLLMReranker(AbsReranker):
    def __init__(
        self,
        model_name_or_path: str = None,
        peft_path: str = None,
        use_fp16: bool = False,
        use_bf16: bool = False,
        cache_dir: str = None,
        trust_remote_code: bool = False,
        device: Union[str, int] = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir,
                                                       trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = 'right'

        if use_bf16 is False and use_fp16 is False:
            warnings.warn("Due to model constraints, `use_bf16` and `use_fp16` cannot both be `False`. Here, `use_fp16` is set to `True` by default.", UserWarning)
            use_fp16 = True

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          cache_dir=cache_dir,
                                                          trust_remote_code=trust_remote_code,
                                                          torch_dtype=torch.bfloat16 if use_bf16 else torch.float32)
        if peft_path:
            self.model = PeftModel.from_pretrained(self.model,peft_path)
            self.model = self.model.merge_and_unload()
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        if device and isinstance(device, str):
            self.device = torch.device(device)
            if device == 'cpu':
                use_fp16 = False
        else:
            if torch.cuda.is_available():
                if device is not None:
                    self.device = torch.device(f"cuda:{device}")
                else:
                    self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False

        if use_fp16 and use_bf16 is False:
            self.model.half()

        self.model = self.model.to(self.device)

        self.model.eval()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 256,
                      max_length: int = 512,
                      cutoff_layers: List[int] = None,
                      compress_layer: List[int] = [8],
                      compress_ratio: int = 1,
                      prompt: str = None,
                      normalize: bool = False,
                      **kwargs: Any) -> List[float]:
    
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-len(q) - len(p) for q, p in sentence_pairs])
        sentences_pairs_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        if prompt is None:
            prompt = "Predict whether passage B contains an answer to query A."
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)['input_ids']
        sep = "\n"
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']
        encode_max_length = max_length + len(sep_inputs) + len(prompt_inputs)
        all_scores = []
        for batch_start in trange(0, len(sentences_pairs_sorted), batch_size):
            batch_sentences = sentences_pairs_sorted[batch_start:batch_start + batch_size]
            batch_sentences = [(f'A: {q}', f'B: {p}') for q, p in batch_sentences]
            queries = [s[0] for s in batch_sentences]
            passages = [s[1] for s in batch_sentences]
            queries_inputs = self.tokenizer(queries,
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length * 3 // 4,
                                            truncation=True,
                                            **kwargs)
            passages_inputs = self.tokenizer(passages,
                                             return_tensors=None,
                                             add_special_tokens=False,
                                             max_length=max_length,
                                             truncation=True,
                                             **kwargs)
            query_lengths = []
            prompt_lengths = []
            batch_inputs = []
            for query_inputs, passage_inputs in zip(queries_inputs['input_ids'], passages_inputs['input_ids']):
                item = self.tokenizer.prepare_for_model(
                    [self.tokenizer.bos_token_id] + query_inputs,
                    sep_inputs + passage_inputs,
                    truncation='only_second',
                    max_length=encode_max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False
                )
                item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                item['attention_mask'] = [1] * len(item['input_ids'])
                item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
                if 'position_ids' in item.keys():
                    item['position_ids'] = list(range(len(item['input_ids'])))
                batch_inputs.append(item)
                query_lengths.append(len([self.tokenizer.bos_token_id] + query_inputs + sep_inputs))
                prompt_lengths.append(len(sep_inputs + prompt_inputs))

            collater_instance = collater_for_lightweight(self.tokenizer, max_length)
            batch_inputs = collater_instance(
                [
                    [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in
                     batch_inputs],
                    query_lengths,
                    prompt_lengths
                ])[0]

            batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}

            outputs = self.model(**batch_inputs,
                                 output_hidden_states=True,
                                 compress_layer=compress_layer,
                                 compress_ratio=compress_ratio,
                                 query_lengths=query_lengths,
                                 prompt_lengths=prompt_lengths,
                                 cutoff_layers=cutoff_layers)
            scores = []
            for i in range(len(outputs.logits)):
                logits = last_logit_pool_lightweight(outputs.logits[i], outputs.attention_masks[i])
                scores.append(logits.cpu().float().tolist())
            if len(all_scores) == 0:
                for i in range(len(scores)):
                    all_scores.append([])
            for i in range(len(scores)):
                all_scores[i].extend(scores[i])

        for i in range(len(all_scores)):
            all_scores[i] = [all_scores[i][idx] for idx in np.argsort(length_sorted_idx)]
            if normalize:
                all_scores[i] = [sigmoid(score) for score in all_scores[i]]

        return all_scores
