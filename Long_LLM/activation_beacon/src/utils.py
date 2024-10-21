import os
import sys
import pytz
import json
import torch
import shutil
import pathlib
import time
import pickle
import logging
import string
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Mapping, Iterable, Union

logger = logging.getLogger(__name__)


@contextmanager
def do_nothing():
    yield

def optional_grad_ctx(with_grad=False):
    if with_grad:
        return do_nothing()
    else:
        return torch.no_grad()

def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

def clear_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def split_file_dir_name_ext(path):
    """Return the directory, name, and extension of a given file."""
    p = pathlib.Path(path)
    assert p.is_file(), f"{path} is not a valid file!"
    return p.parent, p.stem, p.suffix

def save_pickle(obj, path:str):
    """
    Save pickle file.
    """
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def save_json(obj, path:str):
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "w") as f:
        return json.dump(obj, f)

def load_json(path, lines=False):
    if lines:
        output = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                output.append(json.loads(line))
        return output
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def format_numel_str(numel: int) -> str:
    T = 1e12
    B = 1e9
    M = 1e6
    K = 1e3
    if numel >= T:
        return f"{numel / T:.2f} T"
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

def batched_iter(iterable: Iterable, max_batch_size: int):
    """ Batches an iterable into lists of given maximum size, yielding them one by one. """
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= max_batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def show_time(times):
    times = np.array(times)
    times = np.diff(times, axis=-1)
    print(times)
    return times

@contextmanager
def filelock(path, process_index=0):
    while os.path.exists(path):
        if i == 0 and process_index == 0:
            logger.info("found lock, waiting for other programs...")
        time.sleep(3)
        i = 1
    if process_index == 0:
        save_json("this is a lock", path)
    yield
    if process_index == 0:
        os.remove(path)

def normalize_text(text, ignore_case=True, ignore_punctuation=True, ignore_space=True, ignore_number=False):
    if isinstance(text, str):
        text = [text]
        unpack = True
    else:
        unpack = False
    if ignore_case:
        text = np.char.lower(text)
    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        text = np.char.translate(text, table=repl_table)
    if ignore_number:
        repl_table = string.digits.maketrans("", "", string.digits)
        text = np.char.translate(text, table=repl_table)
    if ignore_space:
        for i, words in enumerate(np.char.split(text)):
            text[i] = " ".join(words)
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if unpack:
        text = text[0]
    return text

def wrap_text(s):
    """Capitalize and add punctuation if there isn't."""
    s = s.strip()
    if not s[0].isupper():
        s = s[0].capitalize() + s[1:]
    if s[-1] not in string.punctuation:
        s += "."
    return s

def min_max_normalize(array):
    return (array - array.min(-1)[:,None])/(array.max(-1) - array.min(-1))[:, None]

def softmax(x:np.ndarray, axis=-1):
    if isinstance(x, list):
        x = np.array(x)
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def get_max_length_in_nested_lists(lst):
    if len(lst) and isinstance(lst[0], list):
        lengths = []
        for elem in lst:
            length = get_max_length_in_nested_lists(elem)
            lengths.append(length)
        max_length = max(lengths)
        return max_length
    else:
        return len(lst)

def pad_nested_lists(lst, max_length, padding_value, padding_side="right"):
    if isinstance(lst, list) and len(lst) and isinstance(lst[0], list):
        masks = []
        for i, elem in enumerate(lst):
            lst[i], mask = pad_nested_lists(elem, max_length, padding_value, padding_side)
            masks.append(mask)
        return lst, masks
    elif isinstance(lst, list):
        if padding_side == "right":
            mask = [1] * len(lst) + [0] * (max_length - len(lst))
            lst = lst + [padding_value for _ in range(max_length - len(lst))]
            return lst, mask
        else:
            mask = [0] * (max_length - len(lst)) + [1] * len(lst)
            lst = [padding_value for _ in range(max_length - len(lst))] + lst
            return lst, mask
    else:
        raise NotImplementedError(f"Unrecognized type {lst}")

def mask_nested_lists(lst, mask_target, mask_value=0):
    if isinstance(lst[0], list):
        for i, elem in enumerate(lst):
            lst[i] = mask_nested_lists(elem, mask_target, mask_value)
        return lst
    else:
        return [x if x != mask_target else mask_value for x in lst]

def are_elements_of_same_length(lst: List):
    if not isinstance(lst[0], list):
        return False

    length = len(lst[0])
    return all(len(x) == length if isinstance(x, list) else False for x in lst)

def add_eos(inputs: Mapping, eos_token_id: int):
    """Add eos for BatchEncoding object."""
    assert isinstance(inputs["input_ids"], list), f"Make sure the return_tensors are set to list!"
    if inputs["input_ids"][-1] != eos_token_id:
        for k, v in inputs.items():
            if k in ["input_ids", "labels"]:
                v = v + [eos_token_id]
            elif k == "attention_mask":
                v = v + [1]
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k == "token_type_ids":
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs

def remove_eos(inputs: Mapping, eos_token_ids: Union[List,int]):
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    input_ids = inputs["input_ids"]
    eos_idx = [i for i, x in enumerate(input_ids) if x in eos_token_ids][0]
    for k, v in inputs.items():
        inputs[k].pop(eos_idx)
    return inputs



class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('Asia/Shanghai')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")


@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """
    tokenizer: PreTrainedTokenizer
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_tensorize = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids", "length", "depth", "index"}

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        return_batch = {}
        
        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.tokenizer.pad_token_id

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if isinstance(value, list) and key in self.keys_to_tensorize:
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.tokenizer.padding_side)

            if key in self.keys_to_tensorize and None not in batch_value:
                return_batch[key] = torch.tensor(batch_value)
            else:
                # handle strings and None
                return_batch[key] = batch_value
        return return_batch
