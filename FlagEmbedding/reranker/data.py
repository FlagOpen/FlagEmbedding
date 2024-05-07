import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from arguments import DataArguments


class TrainDatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']
        pos = random.choice(self.dataset[item]['pos'])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        batch_data = []
        batch_data.append(self.create_one_example(query, pos))
        for neg in negs:
            batch_data.append(self.create_one_example(query, neg))

        return batch_data

class TrainDatasetForCL(TrainDatasetForCE):
    def create_one_example(self, input):
        item = self.tokenizer(
            input,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']
        pos = random.choice(self.dataset[item]['pos'])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        batch_data = []
        batch_data.append(self.create_one_example(query))
        batch_data.append(self.create_one_example(pos))
        for neg in negs: batch_data.append(self.create_one_example(neg))
        return batch_data


@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
