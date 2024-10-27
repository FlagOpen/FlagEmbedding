import math
import torch
import random
import datasets
import numpy as np
from glob import glob
from string import Formatter
from typing import Optional, Tuple, Union, List, Callable, Dict, Any, Mapping
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
from transformers.tokenization_utils import PreTrainedTokenizer
from ..utils.util import get_max_length_in_nested_lists, pad_nested_lists, split_file_dir_name_ext, DatasetProcessFn


class RetrievalDataset:
    def get_train_process_fn(train_group_size=8, select_positive="first", select_negative="random", teacher_scores_margin=None, teacher_scores_min=None, stable_distill=False, instruction=None):
        @DatasetProcessFn()
        def _process(query:str, task:str, pos:List[str]=None, neg:List[str]=None, history:List[str]=None, teacher_scores:Optional[List[float]]=None, **kwds):
            output = {}
            keys = []
            if history is not None:
                pos = []
                neg = history

            # filter based on teacher scores
            if teacher_scores is not None:
                assert len(teacher_scores) == len(pos) + len(neg), f"Found incompatible teacher_score size ({len(teacher_scores)}) and positive size ({len(pos)}) negative size ({len(neg)})"
                if teacher_scores_min is not None:
                    max_score = max(teacher_scores)
                    if max_score < teacher_scores_min:
                        return None
                if teacher_scores_margin is not None:
                    max_score = max(teacher_scores)
                    min_score = min(teacher_scores)
                    if max_score - min_score < teacher_scores_margin:
                        return None

            pos_num = len(pos)
            if select_positive == "random":
                assert pos_num > 0, f"Select positive strategy 'random' is only available when there is a given positive!"
                pos_idx = random.choice(range(pos_num))
                pos = pos[pos_idx]
            elif teacher_scores is not None and select_positive == "teacher":
                pos_idx = max(enumerate(teacher_scores), key=lambda x: x[1])[0]
                if pos_idx < pos_num:
                    pos = pos[pos_idx]
                else:
                    # pos is selected from neg, thus we remove it from neg
                    pos = neg.pop(pos_idx - pos_num)
            elif teacher_scores is not None and select_positive == "teacher-pos":
                assert pos_num > 0, f"Select positive strategy 'teacher-pos' is only available when there are teacher_scores and positives!"
                pos_scores = teacher_scores[:pos_num]
                pos_idx = max(enumerate(pos_scores), key=lambda x: x[1])[0]
                pos = pos[pos_idx]
            else:
                # NOTE: default to select the first positive
                assert pos_num > 0, f"Select positive strategy 'first' is only available when there is a given positive!"
                pos_idx = 0
                pos = pos[0]

            if teacher_scores is not None:
                if pos_idx >= pos_num:
                    # only makes sense when select_positive==teacher
                    # remove the selected score
                    pos_score = teacher_scores.pop(pos_idx)
                else:
                    pos_score = teacher_scores[pos_idx]
                # remove teacher scores of unused positives
                neg_scores = teacher_scores[pos_num:]
                return_teacher_scores = [pos_score]

            keys.append(pos)

            if len(neg) == 0:
                return None
            elif len(neg) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(neg))
                neg = neg * num
                if teacher_scores is not None:
                    neg_scores = neg_scores * num

            if teacher_scores is not None and select_negative == "teacher-":
                neg_indices = [i for i, _ in sorted(enumerate(neg_scores), key=lambda x: x[1])[:train_group_size - 1]]
            elif teacher_scores is not None and select_negative == "teacher+":
                neg_indices = [i for i, _ in sorted(enumerate(neg_scores), key=lambda x: x[1], reverse=True)[:train_group_size - 1]]
            elif select_negative == "first":
                neg_indices = list(range(len(neg)))[:train_group_size - 1]
            else:
                # NOTE: default to select random negatives
                neg_indices = random.sample(range(len(neg)), train_group_size - 1)
            for neg_idx in neg_indices:
                keys.append(neg[neg_idx])
                if teacher_scores is not None:
                    return_teacher_scores.append(neg_scores[neg_idx])

            if instruction is not None:
                query = instruction["query"] + query
                keys = [instruction["key"] + key for key in keys]

            output = {
                "query": query,
                "key": keys,
                "task": task,
            }
            if teacher_scores is not None:
                output["teacher_scores"] = return_teacher_scores

            if stable_distill:
                # when using stable_distill, we must sort teacher_scores descendingly
                neg_score = output["teacher_scores"][1:]
                neg = output["key"][1:]
                pairs = sorted(list(zip(neg, neg_score)), key=lambda x: x[1], reverse=True)
                neg = [pair[0] for pair in pairs]
                neg_score = [pair[1] for pair in pairs]
                output["key"][1:] = neg
                output["teacher_scores"][1:] = neg_score

            return output
        return _process

    def prepare_train_dataset(data_file=None, cache_dir=None, config=None, train_group_size=8, select_positive="first", select_negative="random", max_sample_num=None, teacher_scores_margin=None, teacher_scores_min=None, stable_distill=False, add_instruction=False, instruction=None, use_train_config=False):
        if data_file is None:
            return None, None

        if isinstance(data_file, str):
            if "*" in data_file:
                data_file = glob(data_file)
            else:
                data_file = [data_file]

        train_datasets = []
        offset = 0
        dataset_indices_range = {}
        dataset_dup = defaultdict(int)

        for path in data_file:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            task = temp_dataset[0]["task"]
            directory, _, _ = split_file_dir_name_ext(path)
            dataset_name = directory.name

            if add_instruction:
                instruction = config["instruction"][task]
            
            if use_train_config:
                train_config = config["training"][task]
                select_positive = train_config["select_positive"]
                select_negative = train_config["select_negative"]
                max_sample_num = train_config["max_sample_num"]
                teacher_scores_margin = train_config["teacher_scores_margin"]
                teacher_scores_min = train_config["teacher_scores_min"]
                stable_distill = train_config["stable_distill"]

            process_fn = RetrievalDataset.get_train_process_fn(
                train_group_size, 
                select_positive=select_positive,
                select_negative=select_negative,
                teacher_scores_margin=teacher_scores_margin,
                teacher_scores_min=teacher_scores_min,
                stable_distill=stable_distill,
                instruction=instruction
            )
            # map to filter
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=32, remove_columns=temp_dataset.column_names)
            # limit sample number
            if max_sample_num is not None and len(temp_dataset) > max_sample_num:
                temp_dataset = temp_dataset.train_test_split(max_sample_num, shuffle=False)["test"]
            train_datasets.append(temp_dataset)

            if dataset_name in dataset_indices_range:
                # NOTE: we allow duplicated dataset to balance the portion of different datasets
                dataset_dup[dataset_name] += 1
                dataset_indices_range[f"{dataset_name}_{dataset_dup[dataset_name]}"] = (offset, offset + len(temp_dataset))
            else:
                dataset_indices_range[dataset_name] = (offset, offset + len(temp_dataset))
            offset += len(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset, dataset_indices_range
    
    @staticmethod
    def prepare_eval_dataset(data_file=None, cache_dir=None, instruction=None, eval_method="retrieve"):
        if data_file is None:
            return None
        @DatasetProcessFn()
        def _process(query:str, query_id:Optional[int]=None, key:Optional[List[str]]=None, key_index: Optional[List[int]]=None, pos: Optional[List[Union[int, str]]]=None, neg: Optional[List[str]]=None, pos_index:Optional[List[int]]=None, neg_index: Optional[List[int]]=None, _index=None, **kwds):
            if instruction is not None:
                query = instruction["query"] + query
            
            if query_id is None:
                assert _index is not None
                query_id = _index

            output = {
                "query": query,
                "query_id": query_id,
                "task": task,
            }

            if eval_method == "rerank":
                # if there is a column named key, it must be the candidates to rerank
                if key is not None:
                    if key_index is not None:
                        output["key_index"] = key_index
                    else:
                        # NOTE: there must be key_index when reranking
                        output["key_index"] = list(range(len(key)))
                # otherwise, default 
                elif pos is not None and neg is not None:
                    key = pos + neg
                    if pos_index is not None:
                        output["key_index"] = pos_index + neg_index
                    else:
                        # NOTE: there must be key_index when reranking
                        output["key_index"] = list(range(len(key)))
                else:
                    raise ValueError(f"Expected either pos/neg or key in the file {data_file}!")

                if instruction is not None:
                    output["key"] = [instruction["key"] + k for k in key]
                else:
                    output["key"] = key
            return output

        dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)
        if "task" in dataset:
            task = dataset[0]["task"]
        else:
            task = "nan"

        dataset = dataset.map(_process, num_proc=32, batched=True, remove_columns=dataset.column_names, with_indices=True)
        return dataset

    @staticmethod
    def prepare_corpus(data_file, key_template:str, cache_dir=None, instruction=None):
        """Concatenate desired keys by key_template"""
        if data_file is None:
            return None
        keys = Formatter().parse(key_template)
        field_names = [x[1] for x in keys if x[1] is not None]
        @DatasetProcessFn()
        def _process(**kwds):
            inputs = {name: kwds[name] for name in field_names}
            content = key_template.format(**inputs)
            if instruction is not None:
                content = instruction["key"] + content
            return {'content': content}
        dataset = datasets.load_dataset('json', data_files=data_file, split="train", cache_dir=cache_dir)
        dataset.set_transform(_process)
        return dataset


class SameDatasetTrainDataset(torch.utils.data.Dataset):
    """Dataset to yield a batch of data at one time. All samples in the same batch comes from the same task.
    
    Args:
        organize_method: 
            random:
            epoch:
            epoch-random:
            epoch-static
    """
    def __init__(self, dataset, dataset_indices_range, batch_size, seed, organize_method, process_index=0, num_processes=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.organize_method = organize_method
        self.process_index = process_index
        self.num_processes = num_processes

        self.dataset_indices_range = dataset_indices_range

        self.deterministic_generator = np.random.default_rng(seed)
        # different devices must sample different data batch
        self.nondeterministic_generator = np.random.default_rng(seed + process_index)

        # shuffle the indices
        if "random" in self.organize_method:
            self.sample_range = [np.arange(*x) for x in self.dataset_indices_range.values()]
            for x in self.sample_range:
                # NOTE: we must make sure every processes use the same shuffling order
                self.deterministic_generator.shuffle(x)
    
    def create_epoch(self):
        epoch = []
        for k, x in self.dataset_indices_range.items():
            dataset_range = np.arange(*x)
            # NOTE: we must make sure every processes use the same shuffling order
            self.deterministic_generator.shuffle(dataset_range)
            num_batches, remainer = divmod(len(dataset_range), self.batch_size * self.num_processes)
            # Truncate
            if remainer != 0:
                dataset_range = dataset_range[:num_batches * self.batch_size * self.num_processes]

            batches = dataset_range.reshape(num_batches, self.batch_size * self.num_processes).tolist()
            for i in range(len(batches)):
                batches[i] = (k, batches[i])
            epoch.extend(batches)
        # shuffle among datasets, also make sure different processes share the same shuffling results
        self.deterministic_generator.shuffle(epoch)
        self.epoch = epoch
        self.step = 0
        self.steps_per_epoch = len(epoch)

    def __getitem__(self, idx):        
        if self.organize_method == "random":
            sample_prob = [len(x) / len(self.dataset) for x in self.sample_range]

            dataset_name = self.deterministic_generator.choice(range(len(self.sample_range)), size=1, p=sample_prob)[0]
            sample_range = self.sample_range[dataset_name]

            batch_indices = self.nondeterministic_generator.choice(sample_range, size=self.batch_size, replace=False)
            batch_data = self.dataset[batch_indices.tolist()]

        elif self.organize_method == "epoch":
            if not hasattr(self, "epoch") or self.step > self.steps_per_epoch - 1:
                self.create_epoch()

            dataset_name, batch_indices = self.epoch[self.step]
            batch_indices = batch_indices[self.process_index * self.batch_size: (self.process_index + 1) * self.batch_size]
            batch_data = self.dataset[batch_indices]
            self.step += 1
        
        elif self.organize_method == "epoch-static":
            if not hasattr(self, "epoch"):
                # the data within each batch is static once created
                self.create_epoch()
            
            if self.step > self.steps_per_epoch - 1:
                self.deterministic_generator.shuffle(self.epoch)
                self.step = 0

            dataset_name, batch_indices = self.epoch[self.step]
            batch_indices = batch_indices[self.process_index * self.batch_size: (self.process_index + 1) * self.batch_size]
            batch_data = self.dataset[batch_indices]
            self.step += 1
        
        elif self.organize_method == "epoch-random":
            sample_scope = [len(x) for x in self.sample_range]
            sample_prob = [x / sum(sample_scope) for x in sample_scope]

            dataset_name = self.deterministic_generator.choice(range(len(self.sample_range)), size=1, p=sample_prob)[0]
            sample_range = self.sample_range[dataset_name]

            # sequential sample (the indices are already shuffled)
            batch_indices = sample_range[self.process_index * self.batch_size: (self.process_index + 1) * self.batch_size]
            batch_data = self.dataset[batch_indices.tolist()]
            # update indices
            remaining_indices = sample_range[self.num_processes * self.batch_size:]
            if len(remaining_indices) < self.batch_size * self.num_processes:
                remaining_indices = np.array([])
            self.sample_range[dataset_name] = remaining_indices
            # restore all indices if they are all sampled
            if all(len(x) == 0 for x in self.sample_range):
                self.sample_range = [np.arange(*x) for x in self.dataset_indices_range.values()]
                for x in self.sample_range:
                    self.deterministic_generator.shuffle(x)
        else:
            raise NotImplementedError(f"Organize method {self.organize_method} is not implemented for SameTaskTrainDataset!")

        return batch_data
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


@dataclass
class RetrievalDataCollator:
    """
    """
    tokenizer: PreTrainedTokenizer = None
    query_max_length: int = 256
    key_max_length: int = 256
    inbatch_same_dataset: bool = False
    cross: bool = False

    def __call__(self, batch_elem):
        first_elem = batch_elem[0]
        return_batch = {}
        
        for k, v in first_elem.items():
            if self.inbatch_same_dataset:
                # here the data have already been grouped
                batch_value = batch_elem[0][k]
            else:
                batch_value = [elem[k] for elem in batch_elem]
            
            # collate training/evaluating
            if k == "query":
                query = batch_value
                # NOTE: we do not need the individual query and key when requiring cross data
                if self.cross:
                    continue
                batch_value = self.tokenizer(
                    batch_value,
                    padding=True,
                    truncation=True,
                    max_length=self.query_max_length,
                    return_tensors="pt",
                )
            elif k == "key":
                # in case the keys are of different sizes for different queries when reranking
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, key_mask = pad_nested_lists(batch_value, max_length, "", "right")
                batch_value = sum(batch_value, [])
                key = batch_value
                # key_mask assigns 1 to valid keys and 0 to padded keys
                return_batch["key_mask"] = torch.tensor(key_mask)
                # NOTE: we do not need the individual query and key when requiring cross data
                if self.cross:
                    continue
                batch_value = self.tokenizer(
                    batch_value,
                    padding=True,
                    truncation=True,
                    max_length=self.key_max_length,
                    return_tensors="pt",
                )

            elif k == "key_index":
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, -1, "right")
                batch_value = torch.tensor(batch_value)

            elif k == "content":
                # collate corpus
                batch_value = self.tokenizer(
                    batch_value,
                    padding=True,
                    truncation=True,
                    max_length=self.key_max_length,
                    return_tensors="pt",
                )

            elif k == "task":
                assert all(v == batch_value[0] for v in batch_value), f"Make sure all samples are of the same task in a batch!"
                batch_value = batch_value[0]

            elif all(v is None for v in batch_value):
                # in case that some data have teacher_scores but others do not
                batch_value = None

            else:
                batch_value = torch.tensor(batch_value)

            return_batch[k] = batch_value                

        if self.cross:
            query_num = len(query)
            key_num = len(key)
            assert key_num % query_num == 0
            group_size = key_num // query_num
            new_query = []
            for i in range(key_num):
                new_query.append(query[i // group_size])

            return_batch["cross"] = self.tokenizer(
                new_query, key, 
                padding=True, 
                truncation=True,
                max_length=self.key_max_length + self.query_max_length,
                return_tensors="pt"
            )
            return_batch["batch_size"] = len(query)

        return return_batch


TASK_CONFIG = {
    "llm-embedder": {
        "instruction": {
            "qa": {
                "query": "Represent this query for retrieving relevant documents: ",
                "key": "Represent this document for retrieval: ",
            },
            "convsearch": {
                "query": "Encode this query and context for searching relevant passages: ",
                "key": "Encode this passage for retrieval: ",
            },
            "chat": {
                "query": "Embed this dialogue to find useful historical dialogues: ",
                "key": "Embed this historical dialogue for retrieval: ",
            },
            "lrlm": {
                "query": "Embed this text chunk for finding useful historical chunks: ",
                "key": "Embed this historical text chunk for retrieval: ",
            },
            "icl": {
                "query": "Convert this example into vector to look for useful examples: ",
                "key": "Convert this example into vector for retrieval: ",
            },
            "tool": {
                "query": "Transform this user request for fetching helpful tool descriptions: ",
                "key": "Transform this tool description for retrieval: "
            },
        },

        "training": {
            "qa": {
                "select_positive": "first",
                "select_negative": "random",
                "max_sample_num": None,
                "teacher_scores_margin": None,
                "teacher_scores_min": None, 
                "contrastive_weight": 0,
                "stable_distill": True,
            },
            "convsearch": {
                "select_positive": "first",
                "select_negative": "random",
                "max_sample_num": None,
                "teacher_scores_margin": None,
                "teacher_scores_min": None,
                "distill_weight": 0,
                "stable_distill": False,
            },
            "chat": {
                "select_positive": "teacher",
                "select_negative": "random",
                "max_sample_num": None,
                "teacher_scores_margin": None,
                "teacher_scores_min": None,
                "distill_weight": 1.0,
                "contrastive_weight": 0,
                "teacher_temperature": 0.1,
                "stable_distill": False,
            },
            "lrlm": {
                "select_positive": "teacher",
                "select_negative": "random",
                "max_sample_num": 10000,
                "teacher_scores_margin": 0.1,
                "teacher_scores_min": None,
                "distill_weight": 1.0,
                "contrastive_weight": 0,
                "teacher_temperature": 0.1,            
                "stable_distill": False,
            },
            "icl": {
                "select_positive": "random",
                "select_negative": "random",
                "max_sample_num": None,
                "teacher_scores_margin": None,
                "teacher_scores_min": None,
                "contrastive_weight": 0,
                "stable_distill": True,
            },
            "tool": {
                "select_positive": "first",
                "select_negative": "random",
                "max_sample_num": None,
                "teacher_scores_margin": None,
                "teacher_scores_min": None,
                "distill_weight": 0,
                "stable_distill": False,
            },
        }
    },

    "bge": {
        "instruction": defaultdict(lambda: {"query": "Represent this sentence for searching relevant passages: ", "key": ""})
    },
    
    "e5": {
        "instruction": defaultdict(lambda: {"query": "query: ", "key": "passage: "})
    },
    
    "instructor": {
        "instruction": {
            "qa": {
                "query": "Represent the query for retrieving supporting documents: ",
                "key": "Represent the document for retrieval: ",
            },
            "convsearch": {
                "query": "Represent the query and context for retrieving supporting passages: ",
                "key": "Represent the passage for retrieval: ",
            },
            "chat": {
                "query": "Represent the dialogue for retrieving useful historical dialogues: ",
                "key": "Represent the historical dialogue for retrieval: ",
            },
            "lrlm": {
                "query": "Represent the text chunk for retrieving useful historical chunks: ",
                "key": "Represent the historical text chunk for retrieval: ",
            },
            "icl": {
                "query": "Represent the example for retrieving duplicate examples: ",
                "key": "Represent the example for retrieval: ",
            },
            "tool": {
                "query": "Represent the user request for retrieving duplicate examples: ",
                "key": "Represent the tool description for retrieval: "
            },
        },
    }
}
