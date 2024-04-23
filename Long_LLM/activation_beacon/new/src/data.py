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


# RETRIEVAL_CAND = [(1024,1), (512,2), (256,4), (128,8), (512,1), (256,2), (128,4)]
RETRIEVAL_CAND = [(1024,1)]


class Data:
    def _process_language_modeling(data, indices, tokenizer, min_length, max_length):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

        for i, text in enumerate(data['text']):
            # truncate text for faster processing
            # text = text[:max_length * 5]
            # encoded = tokenizer(text, max_length=max_length, truncation=True)
            encoded = tokenizer(text)
            if len(encoded["input_ids"]) < min_length:
                continue
            elif len(encoded['input_ids']) < max_length:
                encoded = add_eos(encoded, tokenizer.eos_token_id)
            else:
                for k, v in encoded.items():
                    encoded[k] = v[:max_length]

            encoded["labels"] = encoded["input_ids"].copy()

            for k, v in encoded.items():
                outputs[k].append(v)
            # length is required for grouping
            outputs["length"].append(len(encoded['input_ids']))
            outputs["index"].append(indices[i])

        return outputs

    def _process_instruction_tuning(data, indices, tokenizer, chat_template, min_length, max_length, eval_mode=False):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

        for i, source in enumerate(data['conversations']):
            if source[0]["role"] != 'user':
                # Skip the first one if it is not from user
                source = source[1:]

            # NOTE: for single-turn conversations, we directly truncate the human side
            # if len(source) == 2:
            #     encoded_user = tokenizer.encode(source[0]['content'])
            #     encoded_assistant = tokenizer.encode(source[1]['content'])
            #     # NOTE: -10 to reserve place for special tokens
            #     user_max_length = max_length - len(encoded_assistant) - 10
            #     if len(encoded_assistant) > max_length:
            #         continue
            #     # NOTE: truncate from the middle
            #     if len(encoded_user) > user_max_length:
            #         half = user_max_length // 2
            #         source[0]['content'] = tokenizer.decode(encoded_user[:half], skip_special_tokens=True) + tokenizer.decode(encoded_user[-half:], skip_special_tokens=True)

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
            )

            # skip data that not fall in between min_length and max_length
            if len(encoded["input_ids"]) < min_length:
                continue
            if len(encoded["input_ids"]) > max_length:
                continue

            if eval_mode:
                encoded["labels"] = labels

            for k, v in encoded.items():
                outputs[k].append(v)
            outputs['length'].append(len(encoded['input_ids']))
            outputs['index'].append(indices[i])

        return outputs

    def _process_needle(data, indices, tokenizer, chat_template, min_length, max_length, retrieval_tuning:float=0, beacon_window:int=1024):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": []}

        for context, needle, question, index in zip(data['context'], data['needle'], data['question'], indices):
            # the needle is wrapped with \n\n
            # the question starts with \n\n
            encoded_needle = tokenizer.encode(needle, add_special_tokens=False)
            encoded_question = tokenizer.encode(question, add_special_tokens=False)
            
            # reserve token for inserting needle, question, answer, and chat template
            context_max_length = max_length - len(encoded_needle) * 2 - len(encoded_question) - 10
            context_min_length = min_length - (max_length - context_max_length)
            encoded_context = tokenizer.encode(context, max_length=context_max_length, truncation=True)
            if len(encoded_context) < context_min_length:
                continue

            needle_pos = random.randint(1, len(encoded_context) - beacon_window)
            
            instruction = tokenizer.decode(encoded_context[:needle_pos] + encoded_needle + encoded_context[needle_pos:] + encoded_question, skip_special_tokens=True)
            answer = needle.strip()

            encoded, conversation = apply_chat_template(
                chat_template, 
                [{'role': 'user', 'content': instruction}, {'role': 'assistant', 'content': answer}], 
                tokenizer=tokenizer,
                return_labels=True,
                return_raw=True,
                max_length=max_length,
                truncation=True
            )

            length = len(encoded['input_ids'])

            if retrieval_tuning > 0:
                do_retrieval_tuning = random.choice([True, False], p=[retrieval_tuning, 1 - retrieval_tuning])
                if do_retrieval_tuning:
                    retrieval_key_length, retrieval_topk = random.choice(RETRIEVAL_CAND)
                    needle_char_pos = conversation.find(needle)
                    needle_pos = len(tokenizer.encode(conversation[:needle_char_pos]))

                    ground_truth_start = needle_pos // retrieval_key_length * retrieval_key_length

                    num_windows = math.ceil(length / beacon_window)

                    starts = [ground_truth_start]
                    if retrieval_topk > 1:
                        candidate_starts = set(range(0, (length - beacon_window) // retrieval_key_length * retrieval_key_length, retrieval_key_length)) - set(starts)
                        starts.extend(random.choice(candidate_starts, size=retrieval_topk - 1, replace=False))

                    spans = []
                    for s in starts:
                        e = s + retrieval_key_length
                        s_floor_window = math.floor(s / beacon_window)
                        e_ceil_window = math.ceil(e / beacon_window)
                        assert e_ceil_window < num_windows, f"Make sure the retrieved span does not overlap with the last window! start_idx {s}, end_idx {e}, start_floor_window {s_floor_window}, end_ceil_window {e_ceil_window}, length {length}, total window number {num_windows}"
                        spans.append((s, e))

                    # sort the spans according to the start_idx
                    spans = sorted(spans, key=lambda x: x[0])

                    if "retrieval_span" not in outputs:
                        outputs["retrieval_span"] = [spans]
                    else:
                        outputs["retrieval_span"].append(spans)

            for k, v in encoded.items():
                outputs[k].append(v)

            # length is required for grouping
            outputs["length"].append(length)
            outputs["index"].append(index)

        return outputs

    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="vicuna", max_train_num_per_data=None, seed=42, retrieval_tuning=0, beacon_window:int=1024, cache_dir=None, load_from_cache_file=None):
        if data_files is None:
            return None

        random.seed(seed)

        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)

            column_names = temp_dataset.column_names
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
            elif "needle" in column_names:
                process_fn = partial(
                    Data._process_needle, 
                    tokenizer=tokenizer, 
                    chat_template=chat_template, 
                    min_length=min_length, 
                    max_length=max_length,
                    retrieval_tuning=retrieval_tuning,
                    beacon_window=beacon_window
                )
            else:
                raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=32, remove_columns=temp_dataset.column_names, batch_size=32, with_indices=True, load_from_cache_file=load_from_cache_file)
            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset

    def prepare_pretrain_data(data_files, tokenizer, config="data/config/default.json", seed=42, cache_dir=None, main_process_first_context=nullcontext, is_main_process=True, load_from_cache_file=None):
        random.seed(seed)

        if isinstance(data_files, list):
            data_files = data_files[0]

        assert os.path.isdir(data_files), f"Make sure the data_files parameter is a directory containing the pretraining data json files! Found {data_files}."
        
        add_eos_token = tokenizer.add_eos_token
        tokenizer.add_eos_token = True
        
        def _process(data):
            input_ids = tokenizer(data["text"])["input_ids"]
            # NOTE: concate all input_ids
            input_ids = sum(input_ids, [])
            return {"input_ids": input_ids}
        
        with open(config, encoding="utf-8") as f:
            config = json.load(f)

        num_instances_per_length = config["num_instances"]
        num_tokens = config["num_tokens_avg"]
        mixture = config["mixture"]

        # concatenate all input_ids and partiton them according to num_instances
        outputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": []}

        for file_name in os.listdir(data_files):
            file_path = os.path.join(data_files, file_name)
            dataset_name = split_file_dir_name_ext(file_path)[1]

            if dataset_name not in mixture:
                continue

            mix_portion = mixture[dataset_name] / 100
            
            if mix_portion == 0:
                continue

            num_instances = {int(k): round(v * mix_portion) for k, v in num_instances_per_length.items()}
            if is_main_process:
                logger.info(f"processing {dataset_name} dataset, {num_instances}")

            # do not need to load all records to meet the requirements in the config
            num_total_tokens = sum(int(k) * v for k, v in num_instances.items())
            # at least load 1 record
            num_records = math.ceil(2 * num_total_tokens / num_tokens[dataset_name])

            with main_process_first_context():
                # tokenize all records
                dataset = datasets.load_dataset("json", data_files=file_path, split=f"train[:{num_records}]", cache_dir=cache_dir)
                dataset = dataset.map(_process, batched=True, num_proc=32, remove_columns=dataset.column_names, batch_size=100, load_from_cache_file=load_from_cache_file)

            instance_length_cand = [k for k, v in num_instances.items() if v > 0]
            instance_length = random.choice(instance_length_cand)

            # endless
            input_ids = dataset["input_ids"]
            cursor = 0

            while cursor + instance_length <= len(dataset):
                instance_input_ids = input_ids[cursor: cursor + instance_length]
                instance_attention_mask = [1 for _ in instance_input_ids]
                instance_labels = instance_input_ids.copy()

                # move the cursor
                cursor += instance_length
                # # NOTE: make sure each chunk starts with a new document
                # while input_ids[cursor] != tokenizer.bos_token_id:
                #     cursor += 1

                # add to final data
                outputs["input_ids"].append(instance_input_ids)
                outputs["attention_mask"].append(instance_attention_mask)
                outputs["labels"].append(instance_labels)
                outputs["length"].append(instance_length)

                num_instances[instance_length] -= 1
                instance_length_cand = [k for k, v in num_instances.items() if v > 0]
                # all needed data have been collected
                if len(instance_length_cand) == 0:
                    break
                elif len(instance_length_cand) == 1:
                    instance_length = instance_length_cand[0]
                else:
                    # re-sample instance length
                    instance_length = random.choice(instance_length_cand)

            if any(v > 0 for v in num_instances.values()):
                logger.warning(f"There are not enough data instances to be sampled! The remainings are {num_instances} for {dataset_name} dataset. Consider increase the corresponding data in {data_files}.")

        dataset = datasets.Dataset.from_dict(outputs)

        tokenizer.add_eos_token = add_eos_token
        return dataset

    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="vicuna", max_eval_num=None, cache_dir=None, seed=42, load_from_cache_file=None):
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
        elif "needle" in column_names:
            process_fn = partial(
                Data._process_needle, 
                tokenizer=tokenizer, 
                chat_template=chat_template, 
                min_length=min_length, 
                max_length=max_length,
            )
        else:
            raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

        dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, with_indices=True, load_from_cache_file=load_from_cache_file)
        return dataset