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


## 3. Train

> **Note**: If you only want to fine-tune the dense embedding of `BAAI/bge-m3`, you can refer to [here](../finetune/README.md).

Here is an simple example of how to perform unified fine-tuning (dense embedding, sparse embedding and colbert) based on `BAAI/bge-m3`:

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
--use_self_distill True
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


For more detailed arguments setting, please refer to [`BGE_M3/arguments.py`](../../FlagEmbedding/BGE_M3/arguments.py).
