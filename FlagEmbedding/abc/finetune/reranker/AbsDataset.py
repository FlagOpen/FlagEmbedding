import os
import math
import random
import logging
import datasets
import numpy as np
import torch.distributed as dist
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    BatchEncoding,
    DataCollatorForSeq2Seq
)
from typing import List

from .AbsArguments import AbsRerankerDataArguments

logger = logging.getLogger(__name__)


class AbsRerankerTrainDataset(Dataset):
    """Abstract class for reranker training dataset.

    Args:
        args (AbsRerankerDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsRerankerDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        self.args = args
        self.tokenizer = tokenizer

        train_datasets = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith('.json') or data_dir.endswith('.jsonl')): continue
                temp_dataset = self._load_dataset(data_dir)
                if len(temp_dataset) == 0: continue
                train_datasets.append(temp_dataset)
            else:
                for file in os.listdir(data_dir):
                    if not (file.endswith('.json') or file.endswith('.jsonl')): continue
                    temp_dataset = self._load_dataset(os.path.join(data_dir, file))
                    if len(temp_dataset) == 0: continue
                    train_datasets.append(temp_dataset)
        self.dataset = datasets.concatenate_datasets(train_datasets)

        self.max_length = self.args.query_max_len + self.args.passage_max_len

    def _load_dataset(self, file_path: str):
        """Load dataset from path.

        Args:
            file_path (str): Path to load the datasets from.

        Raises:
            ValueError: `pos_scores` and `neg_scores` not found in the features of training data

        Returns:
            datasets.Dataset: Loaded HF dataset.
        """
        if dist.get_rank() == 0:
            logger.info(f'loading data from {file_path} ...')

        temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=self.args.cache_path)
        if len(temp_dataset) > self.args.max_example_num_per_dataset:
            temp_dataset = temp_dataset.select(random.sample(list(range(len(temp_dataset))), self.args.max_example_num_per_dataset))
        if not self.args.knowledge_distillation:
            if 'pos_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['pos_scores'])
            if 'neg_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['neg_scores'])
        else:
            if 'pos_scores' not in temp_dataset.column_names or 'neg_scores' not in temp_dataset.column_names:
                raise ValueError(f"`pos_scores` and `neg_scores` not found in the features of training data in {file_path}, which is necessary when using knowledge distillation.")
        return temp_dataset

    def _shuffle_text(self, text):
        """shuffle the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Shuffled text.
        """
        if self.args.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.args.shuffle_ratio:
            split_text = []
            chunk_size = len(text)//3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i+chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def __len__(self):
        return len(self.dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        """Creates a single input example by encoding and preparing a query and document pair for the model.

        Args:
            qry_encoding (str): Query to be encoded.
            doc_encoding (str): Document to be encoded.

        Returns:
            dict: A dictionary containing tokenized and prepared inputs, ready for model consumption.
        """
        qry_inputs = self.tokenizer.encode(qry_encoding, truncation=True, max_length=self.args.query_max_len + self.args.passage_max_len // 4, add_special_tokens=False)
        doc_inputs = self.tokenizer.encode(doc_encoding, truncation=True, max_length=self.args.passage_max_len + self.args.query_max_len // 2, add_special_tokens=False)
        item = self.tokenizer.prepare_for_model(
            qry_inputs,
            doc_inputs,
            truncation='only_second',
            max_length=self.args.query_max_len + self.args.passage_max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        query = data['query']
        if self.args.query_instruction_for_rerank is not None:
            query = self.args.query_instruction_format.format(
                data['query_prompt'] if 'query_prompt' in data else self.args.query_instruction_for_rerank,
                query
            )

        passages = []
        teacher_scores = []

        assert isinstance(data['pos'], list) and isinstance(data['neg'], list)

        pos_idx = random.choice(list(range(len(data['pos']))))
        passages.append(self._shuffle_text(data['pos'][pos_idx]))

        neg_all_idx = list(range(len(data['neg'])))
        if len(data['neg']) < train_group_size - 1:
            num = math.ceil((train_group_size - 1) / len(data['neg']))
            neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
        else:
            neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
        for neg_idx in neg_idxs:
            passages.append(data['neg'][neg_idx])

        if self.args.knowledge_distillation:
            assert isinstance(data['pos_scores'], list) and isinstance(data['neg_scores'], list)
            teacher_scores.append(data['pos_scores'][pos_idx])
            for neg_idx in neg_idxs:
                teacher_scores.append(data['neg_scores'][neg_idx])
            if not all(isinstance(score, (int, float)) for score in teacher_scores):
                raise ValueError(f"pos_score or neg_score must be digit")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_rerank is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    data['passage_prompt'] if 'passage_prompt' in data else self.args.passage_instruction_for_rerank, p
                )
                for p in passages
            ]

        batch_data = []
        for passage in passages:
            batch_data.append(self.create_one_example(query, passage))

        return batch_data, teacher_scores

@dataclass
class AbsRerankerCollator(DataCollatorWithPadding):
    """
    The abstract reranker collator.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features) -> List[BatchEncoding]:
        teacher_scores = [f[1] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        features = [f[0] for f in features]
        if isinstance(features[0], list):
            features = sum(features, [])

        collated = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.query_max_len + self.passage_max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return {
            "pair": collated,
            "teacher_scores": teacher_scores,
        }

class AbsLLMRerankerTrainDataset(AbsRerankerTrainDataset):
    """Abstract class for LLM reranker training dataset.

    Args:
        args (AbsRerankerDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsRerankerDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        super().__init__(args, tokenizer)
        sep = self.args.sep_token
        self.sep_inputs = self.tokenizer(
            sep,
            return_tensors=None,
            add_special_tokens=False
        )['input_ids']

    def __getitem__(self, item) -> List[BatchEncoding]:
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        query = data['query']
        if self.args.query_instruction_for_rerank is not None:
            query = self.args.query_instruction_format.format(
                data['query_prompt'] if 'query_prompt' in data else self.args.query_instruction_for_rerank,
                query
            )

        passages = []
        teacher_scores = []

        assert isinstance(data['pos'], list) and isinstance(data['neg'], list)

        pos_idx = random.choice(list(range(len(data['pos']))))
        passages.append(self._shuffle_text(data['pos'][pos_idx]))

        neg_all_idx = list(range(len(data['neg'])))
        if len(data['neg']) < train_group_size - 1:
            num = math.ceil((train_group_size - 1) / len(data['neg']))
            neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
        else:
            neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
        for neg_idx in neg_idxs:
            passages.append(data['neg'][neg_idx])

        if self.args.knowledge_distillation:
            assert isinstance(data['pos_scores'], list) and isinstance(data['neg_scores'], list)
            teacher_scores.append(data['pos_scores'][pos_idx])
            for neg_idx in neg_idxs:
                teacher_scores.append(data['neg_scores'][neg_idx])
            if not all(isinstance(score, (int, float)) for score in teacher_scores):
                raise ValueError(f"pos_score or neg_score must be digit")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_rerank is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    data['passage_prompt'] if 'passage_prompt' in data else self.args.passage_instruction_for_rerank, p
                )
                for p in passages
            ]

        prompt = self.dataset[item].get('prompt', "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.")

        query_inputs = self.tokenizer(
            query,
            return_tensors=None,
            max_length=self.args.query_max_len + self.args.passage_max_len // 4,
            truncation=True,
            add_special_tokens=False
        )

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors=None,
            add_special_tokens=False
        )['input_ids']

        max_length = self.max_length - len(prompt_inputs) - len(self.sep_inputs)

        passages_inputs = []
        for i, passage in enumerate(passages):
            passage_inputs = self.tokenizer(
                passage,
                return_tensors=None,
                max_length=self.args.passage_max_len + self.args.query_max_len // 2,
                truncation=True,
                add_special_tokens=False
            )
            if self.tokenizer.bos_token_id is not None and self.tokenizer.bos_token_id != self.tokenizer.pad_token_id:
                item = self.tokenizer.prepare_for_model(
                    [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                    self.sep_inputs + passage_inputs['input_ids'],
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False
                )
            else:
                item = self.tokenizer.prepare_for_model(
                    query_inputs['input_ids'],
                    self.sep_inputs + passage_inputs['input_ids'],
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False
                )

            passage_inputs['input_ids'] = item['input_ids'] + self.sep_inputs + prompt_inputs

            passage_inputs['attention_mask'] = [1] * len(passage_inputs['input_ids'])
            # passage_inputs['labels'] = passage_inputs['input_ids'].copy()
            # passage_inputs['labels'] = [-100] * (len(passage_inputs['input_ids']) - 1) + passage_inputs['labels'][(len(passage_inputs['input_ids']) - 1):]
            passage_inputs.pop('token_type_ids') if 'token_type_ids' in passage_inputs.keys() else None
            if 'position_ids' in passage_inputs.keys():
                passage_inputs['position_ids'] = list(range(len(passage_inputs['input_ids'])))
            passages_inputs.append(passage_inputs)

        return passages_inputs, teacher_scores


@dataclass
class AbsLLMRerankerCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        teacher_scores = [f[1] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        features = [f[0] for f in features]
        if isinstance(features[0], list):
            features = sum(features, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            # print(max_label_length)
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
                        feature["labels"] + remainder
                        if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        collated = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.query_max_len + self.passage_max_len,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            "pair": collated,
            "teacher_scores": teacher_scores,
        }
