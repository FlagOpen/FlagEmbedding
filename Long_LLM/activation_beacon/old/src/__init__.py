from .util import FileLogger, DatasetProcessFn, DefaultDataCollator, makedirs, split_file_dir_name_ext, clear_dir, get_max_length_in_nested_lists, pad_nested_lists, mask_nested_lists, are_elements_of_same_length, normalize_text, load_json, save_json, load_pickle, save_pickle, add_eos, remove_eos
from .args import ModelArgs, TrainingArgs
from .data import Data

from .activation_beacon_llama import get_model_and_tokenizer, evaluate_perplexity, evaluate_generation, move_to_device

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

import torch
torch.set_printoptions(linewidth=200)

# import transformers
# transformers.logging.set_verbosity_error()
