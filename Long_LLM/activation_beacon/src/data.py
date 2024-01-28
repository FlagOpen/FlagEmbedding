import datasets
import numpy as np
from glob import glob
from .util import DatasetProcessFn, add_eos


class Data:
    @staticmethod
    def get_process_fn(tokenizer, min_length, max_length, seed=42, with_labels=True):
        patterns = [" ", "\n", "\n\n"]
        rng = np.random.default_rng(seed=seed)
        
        @DatasetProcessFn()
        def process_fn(text=None, input=None, output=None, input_ids=None, labels=None, _index=None, **kwds):
            if text is not None:
                # truncate text for faster processing
                text = text[:max_length * 5]
                inputs = tokenizer(text, max_length=max_length, truncation=True)
                if len(inputs["input_ids"]) < min_length:
                    return None
                inputs["labels"] = inputs["input_ids"].copy()

            elif input is not None:
                input = input.strip()
                output = output.strip()
                # for too long inputs, we truncate first to save encoding time
                if len(input) > 5 * max_length:
                    input = input[:(5 * max_length) // 2] + input[-(5 * max_length) // 2:]

                tokenized_input = tokenizer.encode(input)
                tokenized_output = tokenizer.encode(output, add_special_tokens=False)
                output_length = len(tokenized_output)                
                # some outputs are too long, discard them
                if output_length > max_length:
                    return None

                # truncate from middle
                input_max_length = max_length - output_length
                if len(tokenized_input) > input_max_length:
                    half = int(input_max_length / 2)
                    input = tokenizer.decode(tokenized_input[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_input[-half:], skip_special_tokens=True)
                
                if with_labels:
                    pattern = rng.choice(patterns).tolist()
                    inputs = tokenizer(pattern.join([input, output]))
                    if len(inputs["input_ids"]) < min_length:
                        return None
                    labels = inputs["input_ids"].copy()
                    labels[:-output_length] = [-100 for _ in range(len(labels) - output_length)]
                    inputs["labels"] = labels
                    # NOTE: eos is essential for LLM to learn to stop generation
                    inputs = add_eos(inputs, tokenizer.eos_token_id)
                else:
                    inputs = tokenizer(input)

            elif input_ids is not None:
                if len(input_ids) < min_length or len(input_ids) > max_length:
                    return None
                inputs = {
                    "input_ids": input_ids,
                    "labels": labels,
                }

            else:
                raise NotImplementedError(f"The dataset must contain one of the following fields: 'input' & 'output' (for fine-tuning), 'text' (for pre-training), 'input_ids' & 'labels' for passkey_retrieval.")

            # index is required for evaluation, by default we always add it
            inputs["index"] = _index
            # length is required for grouping
            inputs["length"] = len(inputs["input_ids"])
            return inputs

        return process_fn
    
    @staticmethod
    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, max_eval_num=None, cache_dir=None, seed=42):
        if data_files is None:
            return None

        process_fn = Data.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, seed=seed, with_labels=False)

        if max_eval_num is not None:
            dataset = datasets.load_dataset('json', data_files=data_files, split=f'train[:{max_eval_num}]', cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=data_files, split='train', cache_dir=cache_dir)
        dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, with_indices=True)
        return dataset

    @staticmethod
    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, max_train_num_per_data=None, seed=42, cache_dir=None):
        if data_files is None:
            return None

        if isinstance(data_files, str):
            if "*" in data_files:
                data_files = glob(data_files)
            else:
                data_files = [data_files]
        
        process_fn = Data.get_process_fn(tokenizer, max_length=max_length, min_length=min_length, seed=seed)

        train_datasets = []
        for path in data_files:
            temp_dataset = datasets.load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
            temp_dataset = temp_dataset.map(process_fn, batched=True, num_proc=32, batch_size=1280, remove_columns=temp_dataset.column_names)
            if max_train_num_per_data is not None and len(temp_dataset) > max_train_num_per_data:
                temp_dataset = temp_dataset.train_test_split(max_train_num_per_data, seed=seed)["test"]
            train_datasets.append(temp_dataset)

        dataset = datasets.concatenate_datasets(train_datasets)
        return dataset

