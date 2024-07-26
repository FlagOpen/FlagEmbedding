import os
import json
import random
import math
import datasets
from tqdm import tqdm
from typing import List
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer

from src import split_file_dir_name_ext, get_model_and_tokenizer, format_numel_str, ModelArgs


logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    config: str = field(
        default="data/config/slimpajama.json",
        metadata={'help': 'Configuration json path for standard pretraining (concatenating multiple documents to form instances of equal lengths).'}
    )
    train_data: str = field(
        default="long-llm:slimpajama",
        metadata={'help': 'Directory of training data (multiple json files whose name correspond to the ones in config).'}
    )
    output_dir: str = field(
        default="data/pretrain/llama-8K_2B",
        metadata={'help': 'Output directory for results and logs.'}
    )

    num_token: List[str] = field(
        default_factory=lambda: ["8192:2B"],
        metadata={'help': 'How many tokens to use for a specified length? (T/t for trillion, B/b for billion, M/m for million)'}
    )
    add_bos: bool = field(
        default=True,
        metadata={'help': 'Add bos at the end of each document?'}
    )
    add_eos: bool = field(
        default=True,
        metadata={'help': 'Add eos at the end of each document?'}
    )
    seed: int = field(
        default=123,
        metadata={'help': 'Random seed.'}
    )


def prepare_pretrain_data(data_files, tokenizer: PreTrainedTokenizer, config: dict, length_2_num_token: dict, add_bos:bool=True, add_eos:bool=True, seed=42, cache_dir=None, load_from_cache_file=None):
    random.seed(seed)

    if isinstance(data_files, list):
        data_files = data_files[0]

    assert os.path.isdir(data_files), f"Make sure the data_files parameter is a directory containing the pretraining data json files! Found {data_files}."
        
    def _process(data):
        input_ids = tokenizer(data["text"], add_special_tokens=False)["input_ids"]
        return {"input_ids": input_ids}

    num_token_avg_per_source = config["num_tokens_avg"]
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

        num_token_this_dataset = {k: math.ceil(v * mix_portion) for k, v in length_2_num_token.items()}
        num_instances_this_dataset = {k: math.ceil(v / k) for k, v in num_token_this_dataset.items()}
        info = {k: format_numel_str(v) for k, v in num_token_this_dataset.items()}
        logger.info(f"processing {dataset_name} dataset, generating {info} tokens...")

        # tokenize all records
        dataset = datasets.load_dataset("json", data_files=file_path, split="train", cache_dir=cache_dir)
        dataset = dataset.map(_process, batched=True, num_proc=32, remove_columns=dataset.column_names, batch_size=100, load_from_cache_file=load_from_cache_file)

        tqdm_bar = tqdm(total=sum(num_instances_this_dataset.values()))

        max_length_candidates = [k for k, v in num_instances_this_dataset.items() if v > 0]
        max_length = random.choice(max_length_candidates)

        input_ids = []
        for x in dataset:
            sample_input_ids = x["input_ids"]
            if add_bos:
                assert tokenizer.bos_token_id is not None, f"Make sure the bos_token_id exists when enable add_eos."
                sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids
            if add_eos:
                assert tokenizer.eos_token_id is not None, f"Make sure the eos_token_id exists when enable add_eos."
                sample_input_ids = sample_input_ids + [tokenizer.eos_token_id]
            # add input_ids
            input_ids.extend(sample_input_ids)
            
            if len(input_ids) >= max_length:
                cursor = 0
                while cursor + max_length <= len(input_ids):
                    instance_input_ids = input_ids[cursor: cursor + max_length].copy()
                    instance_attention_mask = [1 for _ in instance_input_ids]
                    instance_labels = instance_input_ids.copy()

                    # move the cursor
                    cursor += max_length

                    # add to final data
                    outputs["input_ids"].append(instance_input_ids)
                    outputs["attention_mask"].append(instance_attention_mask)
                    outputs["labels"].append(instance_labels)
                    outputs["length"].append(max_length)

                    # update num_instances
                    num_instances_this_dataset[max_length] -= 1
                    tqdm_bar.update(1)

                    # sample new max_length
                    max_length_candidates = [k for k, v in num_instances_this_dataset.items() if v > 0]
                    if len(max_length_candidates) == 0:
                        # all needed data have been collected
                        break
                    elif len(max_length_candidates) == 1:
                        max_length = max_length_candidates[0]
                    else:
                        max_length = random.choice(max_length_candidates)

                # remove input_ids that have been saved in outputs
                input_ids = input_ids[cursor:]

            # all needed data have been collected
            if len(max_length_candidates) == 0:
                break

        tqdm_bar.close()

        if len(max_length_candidates) > 0:
            logger.warning(f"There are not enough data ! The remainings are {num_instances_this_dataset} instances for {dataset_name} dataset. Consider increase the corresponding data in {data_files}.")

    dataset = datasets.Dataset.from_dict(outputs)
    
    return dataset



if __name__ == "__main__":
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(days=10))])
    # this script may be executed in DDP, so we make sure the dataset is create only on the main process
    if accelerator.process_index == 0:
        tokenizer = get_model_and_tokenizer(args, return_tokenizer_only=True)

        if args.add_eos:
            assert tokenizer.eos_token_id is not None, "Make sure the eos_token_id is not None when enabling add_eos!"

        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)

        length_2_num_token = {}
        for x in args.num_token:
            length, ntok = x.split(":")
            length = int(length)

            if ntok.lower().endswith("t"):
                ntok = float(ntok[:-1]) * 1e12
            elif ntok.lower().endswith("b"):
                ntok = float(ntok[:-1]) * 1e9
            elif ntok.lower().endswith("m"):
                ntok = float(ntok[:-1]) * 1e6
            else:
                raise ValueError(f"Make sure num_token ends with T/t/B/b/M/m!")

            length_2_num_token[length] = ntok

        pretrain_dataset = prepare_pretrain_data(
            args.train_data, 
            tokenizer=tokenizer,
            config=config,
            length_2_num_token=length_2_num_token,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            seed=args.seed,
            cache_dir=args.dataset_cache_dir,
        )
        
        logger.info(f"Saving dataset to {args.output_dir}...")
        pretrain_dataset.save_to_disk(args.output_dir)

    accelerator.wait_for_everyone()
