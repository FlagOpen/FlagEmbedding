import re
import os
import json
import math
import random
import datasets
from tqdm import tqdm
from functools import partial
from glob import glob
from contextlib import nullcontext
from transformers.utils import logging
from src import apply_chat_template, add_eos, split_file_dir_name_ext

logger = logging.get_logger(__name__)



class Data:
    def _process_pretrain_data(data, indices):
        outputs = {"labels": [], "index": [], "length": []}
        for input_ids, index in zip(data['input_ids'], indices):
            outputs["index"].append(index)
            outputs["length"].append(len(input_ids))
            # NOTE: the labels will be automatically generated in Trainer._prepare_inputs
            outputs["labels"].append(None)
        return outputs

    def _process_language_modeling(data, indices, tokenizer, min_length, max_length):
        outputs = {'input_ids': [], "labels": [], "length": [], "index": []}

        for i, text in enumerate(data['text']):
            # truncate text for faster processing
            encoded = tokenizer(text)
            if len(encoded["input_ids"]) < min_length:
                continue
            elif len(encoded['input_ids']) < max_length:
                encoded = add_eos(encoded, tokenizer.eos_token_id)
            else:
                for k, v in encoded.items():
                    encoded[k] = v[:max_length]

            # NOTE: the labels will be automatically generated in Trainer._prepare_inputs
            encoded["labels"] = None

            for k, v in encoded.items():
                if k in outputs:
                    outputs[k].append(v)
            # length is required for grouping
            outputs["length"].append(len(encoded['input_ids']))
            outputs["index"].append(indices[i])

        return outputs

    def _process_instruction_tuning(data, indices, tokenizer, chat_template, min_length, max_length, eval_mode=False):
        outputs = {'input_ids': [], "labels": [], "length": [], "index": []}

        for i, source in enumerate(data['conversations']):
            if source[0]["role"] != 'user':
                # Skip the first one if it is not from user
                source = source[1:]

            # NOTE: in evaluation, we only use the first turn in the conversation
            if eval_mode:
                # a string (the expected output from the assistant)
                if len(source) > 1:
                    labels = source[1]['content']
                else:
                    labels = None
                source = source[:1]

            encoded = apply_chat_template(
                chat_template, 
                source, 
                tokenizer=tokenizer, 
                # only return labels in evaluation mode
                return_labels=not eval_mode,
                add_generation_prompt=eval_mode, 
            ).encoded

            # NOTE: shift the labels in advance
            # labels = encoded["labels"][1:]
            # labels.append(-100)
            # encoded["labels"] = labels

            # skip data that not fall in between min_length and max_length
            if min_length is not None and len(encoded["input_ids"]) < min_length:
                continue
            if max_length is not None and len(encoded["input_ids"]) > max_length:
                continue

            if eval_mode:
                encoded["labels"] = labels

            for k, v in encoded.items():
                if k in outputs:
                    outputs[k].append(v)
            outputs['length'].append(len(encoded['input_ids']))
            outputs['index'].append(indices[i])

        return outputs

    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="vicuna", seed=42, cache_dir=None, load_from_cache_file=None, ignore_index=False, ignore_length=False):
        if data_files is None:
            return None

        if isinstance(data_files, list):
            logger.info(f"Loading training data from {data_files}...")
        elif isinstance(data_files, str):
            logger.info(f"Loading training data from {data_files}...")
            data_files = [data_files]
        else:
            raise ValueError(f"Invalid training data {data_files}!")

        data_2_num_sample = {}
        for data_file in data_files:
            match = re.search("\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub("\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            data_2_num_sample[data_file] = max_sample_num   
        
        random.seed(seed)
        
        train_datasets = []
        for data_file, max_sample_num in data_2_num_sample.items():

            if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
                # the dataset may be save_to_disk in advance
                dataset = datasets.load_from_disk(data_file)
                dataset = dataset.map(Data._process_pretrain_data, batched=True, num_proc=32, batch_size=32, with_indices=True)

            else:
                # the dataset is a json file
                dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)

                column_names = dataset.column_names
                if "text" in column_names:
                    process_fn = partial(
                        Data._process_language_modeling, 
                        tokenizer=tokenizer, 
                        min_length=min_length, 
                        max_length=max_length
                    )
                elif "conversations" in column_names:
                    process_fn = partial(
                        Data._process_instruction_tuning, 
                        tokenizer=tokenizer, 
                        chat_template=chat_template, 
                        min_length=min_length, 
                        max_length=max_length
                    )
                else:
                    raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

                dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, batch_size=32, with_indices=True, load_from_cache_file=load_from_cache_file)

            if max_sample_num is not None and len(dataset) > max_sample_num:
                dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

            # index column is useless in training
            if "index" in dataset.column_names and ignore_index:
                dataset = dataset.remove_columns(["index"])
            if "length" in dataset.column_names and ignore_length:
                dataset = dataset.remove_columns(["length"])

            train_datasets.append(dataset)

        dataset = datasets.concatenate_datasets(train_datasets)

        return dataset

    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="vicuna", max_eval_num=None, cache_dir=None, seed=42, load_from_cache_file=None, ignore_index=False, ignore_length=False):
        if data_files is None:
            return None

        random.seed(seed)

        if max_eval_num is not None:
            dataset = datasets.load_dataset('json', data_files=data_files, split=f'train[:{max_eval_num}]', cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=data_files, split='train', cache_dir=cache_dir)

        column_names = dataset.column_names
        if "text" in column_names:
            process_fn = partial(
                Data._process_language_modeling, 
                tokenizer=tokenizer, 
                min_length=min_length, 
                max_length=max_length
            )
        elif "conversations" in column_names:
            process_fn = partial(
                Data._process_instruction_tuning, 
                tokenizer=tokenizer, 
                chat_template=chat_template, 
                min_length=min_length, 
                max_length=max_length,
                eval_mode=True,
            )
        else:
            raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

        dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, with_indices=True, load_from_cache_file=load_from_cache_file)
        if "index" in dataset.column_names and ignore_index:
            dataset = dataset.remove_columns(["index"])
        if "length" in dataset.column_names and ignore_length:
            dataset = dataset.remove_columns(["length"])

        return dataset