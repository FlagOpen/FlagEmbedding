import re
import sys
from typing import List, Tuple

import math
import os.path
import random
from dataclasses import dataclass

import datasets
import numpy as np
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, BatchEncoding
from transformers import PreTrainedTokenizer, BatchEncoding

from arguments import DataArguments

def traverse_directory_using_os(root_folder):
    file_list = []
    if not os.path.isdir(root_folder):
        file_list.append(root_folder)
    else:
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                file_list.append(full_path)
    return file_list

class TrainDatasetForReranker(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.exists(args.train_data[-1]):
            train_datasets = []
            data_path = []
            for data_dir in args.train_data:
                data_path.extend(traverse_directory_using_os(data_dir))
            for file in data_path:
                try:
                    temp_dataset = datasets.load_dataset('json', data_files=file,
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
            self.dataset = datasets.load_dataset(args.train_data[-1], split='train', cache_dir=args.cache_path)


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

    def __getitem__(self, item) -> tuple[List[BatchEncoding], List[int], List[int], List[int]]:
        query = self.dataset[item]['query']

        passages = []
        pos = random.choice(self.dataset[item]['pos'])

        try:
            scores = [self.dataset[item]['pos_scores'][self.dataset[item]['pos'].index(pos)]]
        except:
            scores = [1]

        passages.append(pos)
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        for neg in negs:
            try:
                scores.append(self.dataset[item]['neg_scores'][self.dataset[item]['neg'].index(neg)])
            except:
                scores.append(1)

        if self.dataset[item].get('prompt') is not None:
            prompt = self.dataset[item]['prompt']
        else:
            # prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            prompt = "Predict whether passage B contains an answer to query A."

        query = f'{self.args.query_instruction_for_retrieval}{query}'
        passages = [f'{self.args.passage_instruction_for_retrieval}{p}' for p in passages]

        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      # max_length=self.args.query_max_len + self.args.passage_max_len // 4,
                                      # max_length=32,
                                      # padding='max_length',
                                      max_length=self.args.query_max_len,
                                      truncation=True,
                                      add_special_tokens=False)

        positive_inputs = self.tokenizer(prompt,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        max_length = self.max_length - len(positive_inputs) - len(self.sep_inputs)

        passages_inputs = []
        query_inputs_length = []
        prompt_inputs_length = []
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
                query_inputs_length.append(len([self.tokenizer.bos_token_id] + query_inputs['input_ids'] + self.sep_inputs))
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
                query_inputs_length.append(len(query_inputs['input_ids'] + self.sep_inputs))

            passage_inputs['input_ids'] = item['input_ids'] + self.sep_inputs + positive_inputs
            prompt_inputs_length.append(len(self.sep_inputs + positive_inputs))
            passage_inputs['attention_mask'] = [1] * len(passage_inputs['input_ids'])
            passage_inputs.pop('token_type_ids') if 'token_type_ids' in passage_inputs.keys() else None
            if 'position_ids' in passage_inputs.keys():
                passage_inputs['position_ids'] = list(range(len(passage_inputs['input_ids'])))
            passages_inputs.append(passage_inputs)

        return passages_inputs, query_inputs_length, prompt_inputs_length, scores


@dataclass
class RerankCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features_lengths, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        features = [e[0] for e in features_lengths]
        query_lengths = [e[1] for e in features_lengths]
        prompt_lengths = [e[2] for e in features_lengths]
        scores = [e[3] for e in features_lengths]
        if isinstance(features[0], list):
            features = sum(features, [])
        if isinstance(query_lengths[0], list):
            query_lengths = sum(query_lengths, [])
        if isinstance(prompt_lengths[0], list):
            prompt_lengths = sum(prompt_lengths, [])
        if isinstance(scores[0], list):
            scores = sum(scores, [])

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

        return {"pair": collated, "query_lengths": query_lengths, "prompt_lengths": prompt_lengths,
                "teacher_scores": scores}
        # return collated