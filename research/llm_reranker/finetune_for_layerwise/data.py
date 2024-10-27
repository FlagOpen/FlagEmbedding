import re
import sys
from typing import List

import math
import os.path
import random
from dataclasses import dataclass

import datasets
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments


class TrainDatasetForReranker(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                try:
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                         split='train',
                                                         cache_dir=args.cache_path)
                except Exception as e:
                    print(e)
                    print(file)
                    sys.exit()
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)

            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train', cache_dir=args.cache_path)


        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

        sep = "\n"
        self.sep_inputs = self.tokenizer(sep,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        self.max_length = self.args.query_max_len + self.args.passage_max_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        prompt = self.dataset[item]['prompt']

        query = f'{self.args.query_instruction_for_retrieval}{query}'
        passages = [f'{self.args.passage_instruction_for_retrieval}{p}' for p in passages]

        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      max_length=self.args.query_max_len + self.args.passage_max_len // 4,
                                      truncation=True,
                                      add_special_tokens=False)

        positive_inputs = self.tokenizer(prompt,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids'] + \
                          self.tokenizer('Yes',
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        max_length = self.max_length - len(positive_inputs) - len(self.sep_inputs)

        passages_inputs = []
        for i, passage in enumerate(passages):
            passage_inputs = self.tokenizer(passage,
                                            return_tensors=None,
                                            max_length=self.args.passage_max_len + self.args.query_max_len // 2,
                                            truncation=True,
                                            add_special_tokens=False)
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
            passage_inputs['input_ids'] = item['input_ids'] + self.sep_inputs + positive_inputs

            passage_inputs['attention_mask'] = [1] * len(passage_inputs['input_ids'])
            passage_inputs['labels'] = passage_inputs['input_ids'].copy()
            passage_inputs['labels'] = [-100] * (len(passage_inputs['input_ids']) - 1) + passage_inputs['labels'][(len(passage_inputs['input_ids']) - 1):]
            passage_inputs.pop('token_type_ids') if 'token_type_ids' in passage_inputs.keys() else None
            if 'position_ids' in passage_inputs.keys():
                passage_inputs['position_ids'] = list(range(len(passage_inputs['input_ids'])))
            passages_inputs.append(passage_inputs)

        return passages_inputs

@dataclass
class RerankCollator(DataCollatorForSeq2Seq):
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

        if isinstance(features[0], list):
            features = sum(features, [])

        # print(features)

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
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
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

        return {"pair": collated}
        # return collated