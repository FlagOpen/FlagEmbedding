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


def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)


class DatasetForReranker(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer_path: str,
            max_len: int = 512,
            query_prefix: str = 'A: ',
            passage_prefix: str = 'B: ',
            cache_dir: str = None,
            prompt: str = None,
            **kwargs: Any, 
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        self.dataset = dataset
        self.max_len = max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.total_len = len(self.dataset)
        self.kwargs = kwargs

        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        self.prompt_inputs = self.tokenizer(prompt,
                                            return_tensors=None,
                                            add_special_tokens=False)['input_ids']
        sep = "\n"
        self.sep_inputs = self.tokenizer(sep,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        self.encode_max_length = self.max_len + len(self.sep_inputs) + len(self.prompt_inputs)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query, passage = self.dataset[item]
        query = self.query_prefix + query
        passage = self.passage_prefix + passage
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      add_special_tokens=False,
                                      max_length=self.max_len * 3 // 4,
                                      truncation=True,
                                      **self.kwargs)
        passage_inputs = self.tokenizer(passage,
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=self.max_len,
                                        truncation=True,
                                        **self.kwargs)
        item = self.tokenizer.prepare_for_model(
            [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
            self.sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=self.encode_max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + self.sep_inputs + self.prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
        if 'position_ids' in item.keys():
            item['position_ids'] = list(range(len(item['input_ids'])))

        return item

class collater():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8
        self.label_pad_token_id = -100
        warnings.filterwarnings("ignore",
                                message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy.")

    def __call__(self, data):
        labels = [feature["labels"] for feature in data] if "labels" in data[0].keys() else None
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
            for feature in data:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        return self.tokenizer.pad(
            data,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )


class BaseLLMReranker(AbsReranker):
    def __init__(
        self,
        model_name_or_path: str = None,
        peft_path: str = None,
        use_fp16: bool = False,
        use_bf16: bool = False,
        cache_dir: str = None,
        trust_remote_code: bool = False,
        devices: Union[str, List[str], List[int]] = None, # specify devices, such as ["cuda:0"] or ["0"]
        **kwargs: Any,
    ) -> None:

        super().__init__(
            model_name_or_path,
            use_fp16,
            devices,
            **kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir,
                                                       trust_remote_code=trust_remote_code)

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

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        max_length: int = 512,
        prompt: str = None,
        normalize: bool = False,
        use_dataloader: bool = False,
        num_workers: int = None,
        device: str = None,
        **kwargs: Any
    ) -> List[float]:
        
        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()
    
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-len(q) - len(p) for q, p in sentence_pairs])
        sentences_pairs_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        dataset, dataloader = None, None
        if use_dataloader:
            if num_workers is None:
                num_workers = min(batch_size, 16)
            dataset = DatasetForReranker(sentences_pairs_sorted,
                                         self.model_name_or_path,
                                         max_length,
                                         cache_dir=self.cache_dir,
                                         prompt=prompt,
                                         **kwargs)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False,
                                    num_workers=num_workers,
                                    collate_fn=collater(self.tokenizer, max_length))

        all_scores = []
        if dataloader is not None:
            for inputs in tqdm(dataloader):
                inputs = inputs.to(device)

                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                scores = last_logit_pool(logits, inputs['attention_mask'])
                scores = scores[:, self.yes_loc]
                all_scores.extend(scores.cpu().float().tolist())
        else:
            if prompt is None:
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            prompt_inputs = self.tokenizer(prompt,
                                           return_tensors=None,
                                           add_special_tokens=False)['input_ids']
            sep = "\n"
            sep_inputs = self.tokenizer(sep,
                                             return_tensors=None,
                                             add_special_tokens=False)['input_ids']
            encode_max_length = max_length + len(sep_inputs) + len(prompt_inputs)
            for batch_start in trange(0, len(sentences_pairs_sorted), batch_size):
                batch_sentences = sentences_pairs_sorted[batch_start:batch_start + batch_size]
                batch_sentences = [(f'A: {q}', f'B: {p}') for q,p in batch_sentences]
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

                collater_instance = collater(self.tokenizer, max_length)
                batch_inputs = collater_instance(
                    [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in
                     batch_inputs])

                batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}

                outputs = self.model(**batch_inputs, output_hidden_states=True)
                logits = outputs.logits
                scores = last_logit_pool(logits, batch_inputs['attention_mask'])
                scores = scores[:, self.yes_loc]
                all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        # if len(all_scores) == 1:
        #     return all_scores[0]

        return all_scores
