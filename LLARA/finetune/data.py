import sys

import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, BatchEncoding

from arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
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

        self.prefix = '"'
        self.prefix_ids = self.tokenizer(self.prefix, return_tensors=None)['input_ids']
        self.suffix_passage = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
        self.suffix_passage_ids = self.tokenizer(self.suffix_passage, return_tensors=None, add_special_tokens=False)['input_ids']
        # self.suffix_query = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
        self.suffix_query = '", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>'
        self.suffix_query_ids = self.tokenizer(self.suffix_query, return_tensors=None, add_special_tokens=False)['input_ids']
        self.query_max_len = self.args.query_max_len - len(self.prefix_ids) - len(self.suffix_query_ids)
        self.passage_max_len = self.args.passage_max_len - len(self.prefix_ids) - len(self.suffix_passage_ids)

    def __len__(self):
        # return self.total_len
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        query = self.dataset[item]['query']
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      max_length=self.query_max_len,
                                      truncation=True,
                                      add_special_tokens=False)
        query_inputs['input_ids'] = self.prefix_ids + query_inputs['input_ids'] + self.suffix_query_ids
        query_inputs['attention_mask'] = [1] * len(query_inputs['input_ids'])

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            # print(len(self.dataset[item]['neg']))
            num = math.ceil((self.args.train_group_size - 1) / len(list(set(self.dataset[item]['neg']))))
            # negs = random.sample(list(set(self.dataset[item]['neg'])) * num, self.args.train_group_size - 1)
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
            # negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
            # negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        passages_inputs = []
        for passage in passages:
            passage_inputs = self.tokenizer(passage,
                                            return_tensors=None,
                                            max_length=self.passage_max_len,
                                            truncation=True,
                                            add_special_tokens=False)
            passage_inputs['input_ids'] = self.prefix_ids + passage_inputs['input_ids'] + self.suffix_passage_ids
            passage_inputs['attention_mask'] = [1] * len(passage_inputs['input_ids'])
            passages_inputs.append(passage_inputs)

        return query_inputs, passages_inputs


@dataclass
class EmbedCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1

    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        queries = []
        passages = []
        for e in features:
            queries.append(e[0])
            passages.extend(e[1])

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                queries,
                padding=self.padding,
                max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            d_collated = self.tokenizer.pad(
                passages,
                padding=self.padding,
                max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
        else:
            batch_size = self.sub_batch_size

            q_collated = []
            for i in range(0, len(queries), batch_size):
                start = i
                end = min(len(queries), i + batch_size)
                sub_features = queries[start:end]
                q_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                ))

            d_collated = []
            for i in range(0, len(passages), batch_size):
                start = i
                end = min(len(passages), i + batch_size)
                sub_features = passages[start: end]
                d_collated.append(self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=return_tensors,
                ))

        return {"query": q_collated, "passage": d_collated}