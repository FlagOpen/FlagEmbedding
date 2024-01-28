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
from collections import defaultdict, OrderedDict
from typing import Optional, Tuple, Union, List, Callable, Dict, Any, Mapping

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
        return json.dump(obj, f, ensure_ascii=False)

def load_json(path, lines=False):
    if lines:
        output = []
        with open(path, "r") as f:
            for line in f:
                output.append(json.loads(line))
        return output
    else:
        with open(path, "r") as f:
            return json.load(f)

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
            elif k == "position_ids":
                v = v + [v[-1] + 1]
            elif k in ["attention_mask", "token_type_ids"]:
                v = v + v[-1:]
            else:
                raise NotImplementedError(f"Inputs key {k} not implemented!")
            inputs[k] = v
    return inputs

def remove_eos(inputs: Mapping, eos_token_id: int):
    input_ids = inputs["input_ids"]
    eos_idx = [i for i, x in enumerate(input_ids) if x == eos_token_id][0]
    for k, v in inputs.items():
        inputs[k].pop(eos_idx)
    return inputs

def mix_parameters(models: List[torch.nn.Module], weights: Optional[List[float]]=None):
    """Mix parameters of different models according to given weights.
    
    Returns:
        the model with mixed parameters.
    """
    new_state_dict = OrderedDict()
    if weights is None:
        weights = [1 / len(models) for _ in range(len(models))]
    else:
        assert len(weights) == len(models), f"Make sure the size of mix weights equals to the number of models!"

    for name_param_pairs in zip(*[model.state_dict().items() for model in models]):
        names = [name_param_pair[0] for name_param_pair in name_param_pairs]
        params = [name_param_pair[1] for name_param_pair in name_param_pairs]

        assert all(name == names[0] for name in names), f"Found incompatible key in {names}!"
        name = names[0]
        mixed_param = None

        # there may be non-float parameters stored, which should not be mixed
        if params[0].dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            assert all((param == params[0]).all() for param in params), f"Found incompatible value in non-float tensor {params}!"
            new_state_dict[name] = params[0]
            continue

        for weight, param in zip(weights, params):
            if mixed_param is None:
                mixed_param = weight * param
            else:
                mixed_param += weight * param
            new_state_dict[name] = mixed_param
            
    model = models[0]
    info = model.load_state_dict(new_state_dict)
    print(info)
    return model


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


class DatasetProcessFn:
    """Wrapper for any user-defined process function for huggingface datasets.

    1. Process batched examples by looping the process function over them;
    2. Gather returned examples if any data augmentation happens with augment=True;
    3. Pass indices of examples inside the process function with _index keywords if they exist.

    The wrapped function should take in any needed columns and return a dict with 1 or more samples.
    """
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, _process_fn):
        def process(*args):
            sample_or_batch_sample = args[0]
            if len(args) == 1:
                pass
            elif len(args) == 2:
                indices = args[1]
                # detach the slice so that _index will not be set in the original data
                sample_or_batch_sample = sample_or_batch_sample.copy()
                sample_or_batch_sample["_index"] = indices
            else:
                raise NotImplementedError(f"Found more than 2 arguments {args}!")

            keys = list(sample_or_batch_sample.keys())
            func_args = [sample_or_batch_sample[k] for k in keys]
            
            # FIXME: if all values in one sample are of the same length, this would fail
            if are_elements_of_same_length(func_args):
                outputs = defaultdict(list)
                for arg in zip(*func_args):
                    # get each element in a batch
                    kwargs = {keys[j]: arg[j] for j in range(len(arg))}
                    output = _process_fn(**kwargs)
                    if output is not None:
                        for k, v in output.items():
                            if self.augment:
                                outputs[k].extend(v)
                            else:
                                outputs[k].append(v)
            else:
                outputs = _process_fn(**sample_or_batch_sample)
                if outputs is None:
                    raise ValueError(f"Found None returned from process_fn. Make sure you set 'batched=True' when trying to augment/distract samples in the datasets!")
            return dict(outputs)
        return process


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
    add_position_ids: bool = False

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
            if isinstance(value, list):
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.tokenizer.padding_side)

            try:
                return_batch[key] = torch.tensor(batch_value)
            except:
                # handle strings and None
                return_batch[key] = batch_value

            if "attention_mask" in key and self.add_position_ids:
                value = return_batch[key]
                position_ids = value.cumsum(-1) - 1
                position_ids = position_ids.masked_fill(value == 0, 0)
                return_batch[key.replace("attention_mask", "position_ids")] = position_ids
        return return_batch
