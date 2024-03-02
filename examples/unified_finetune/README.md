# Unified Finetune

In this example, we show how to perform unified fine-tuning based on [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) with your data.

## 1. Installation

- with pip

```
pip install -U FlagEmbedding
```

- from source

```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install .
```
For development, install as editable:

```
pip install -e .
```

## 2. Data format

Training data should be a jsonl file, where each line is a dict like this:

```
{"query": str, "pos": List[str], "neg":List[str]}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts.

If you want to use knowledge distillation, each line of your jsonl file should be like this:

```
{"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[float], "neg_scores": List[float]}
```

`pos_scores` is a list of positive scores, where `pos_scores[i]` is the score between `query` and `pos[i]` from the teacher model. `neg_scores` is a list of negative scores, where `neg_scores[i]` is the score between `query` and `neg[i]` from the teacher model.

See [toy_train_data](./toy_train_data) for an example of training data.

### Use efficient batching strategy [Optional]

(*Optional*) If you want to use **efficient batching strategy** (for more details, please refer to [our paper](https://arxiv.org/pdf/2402.03216.pdf)), you should use this [script](../../BGE_M3/split_data_by_length.py) to split your data to different parts by sequence length before training. Here's an example of how to use this script to split your data to different parts by sequence length:

```bash
python split_data_by_length.py \
--input_path train_data \
--output_dir train_data_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \
--overwrite False
```

`input_path` is the path of jsonl file or the directory containing some jsonl files. `output_dir` is the directory where the split files (`*_len-0-500.jsonl`, `*_len-500-1000.jsonl`, etc.) are saved. `output_dir/log_name` is the log of split data. `length_list` is the list of sequence length. `model_name_or_path` is used to tokenize the data. `num_proc` is the number of processes to use. `overwrite` is whether to overwrite the existing files.

For example, if there are two jsonl files `train_data1.jsonl` and `train_data2.jsonl` in `train_data` directory, then after running the above script, there will be some split files in `train_data_split` like this:

```
train_data_split
├── train_data1_0-500.jsonl
├── train_data1_500-1000.jsonl
├── train_data1_1000-2000.jsonl
├── train_data1_2000-3000.jsonl
├── train_data1_3000-4000.jsonl
├── train_data1_4000-5000.jsonl
├── train_data1_5000-6000.jsonl
├── train_data1_6000-7000.jsonl
├── train_data1_7000-inf.jsonl
├── train_data2_0-500.jsonl
├── train_data2_500-1000.jsonl
├── train_data2_1000-2000.jsonl
├── train_data2_2000-3000.jsonl
├── train_data2_3000-4000.jsonl
├── train_data2_4000-5000.jsonl
├── train_data2_5000-6000.jsonl
├── train_data2_6000-7000.jsonl
├── train_data2_7000-inf.jsonl
```

Note that if there's no data in a specific range, the corresponding file will not be created.

## 3. Train

> **Note**: If you only want to fine-tune the dense embedding of `BAAI/bge-m3`, you can refer to [here](../finetune/README.md).

Here is an simple example of how to perform unified fine-tuning based on `BAAI/bge-m3`:

```bash
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.BGE_M3.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-m3 \
--train_data ./toy_train_data \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--same_task_within_batch True \
--unified_finetuning True \
--use_self_distill True \
--fix_encoder False
```

You can also refer to [this script](./unified_finetune_bge-m3_exmaple.sh) for more details. In this script, we use `deepspeed` to perform distributed training. Learn more about `deepspeed` at https://www.deepspeed.ai/getting-started/. Note that there are some important parameters to be modified in this script:

- `HOST_FILE_CONTENT`: Machines and GPUs for training. If you want to use multiple machines for training, please refer to https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node (note that you should configure `pdsh` and `ssh` properly).
- `DS_CONFIG_FILE`: Path of deepspeed config file. [Here](../finetune/ds_config.json) is an example of `ds_config.json`.
- `DATA_PATH`: One or more paths of training data. **Each path must be a directory containing one or more jsonl files**.
- `DEFAULT_BATCH_SIZE`: Default batch size for training. If you use efficient batching strategy, which means you have split your data to different parts by sequence length, then the batch size for each part will be decided by the `get_file_batch_size()` function in [`BGE_M3/data.py`](../../FlagEmbedding/BGE_M3/data.py). Before starting training, you should set the corresponding batch size for each part in this function according to the GPU memory of your machines. `DEFAULT_BATCH_SIZE` will be used for the part whose sequence length is not in the `get_file_batch_size()` function.
- `EPOCHS`: Number of training epochs.
- `LEARNING_RATE`: The initial learning rate.
- `SAVE_PATH`: Path of saving finetuned model.

 You should set these parameters appropriately.


For more detaild arguments setting, please refer to [`BGE_M3/arguments.py`](../../FlagEmbedding/BGE_M3/arguments.py).
