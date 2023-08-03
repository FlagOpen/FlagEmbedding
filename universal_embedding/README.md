# Finetune

## Installation
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .
```
For development, install as editable:
```
pip install -e .
```
 

## Pre-train


#### 1. Data format
Train data should be a json file, where each line is a dict like this:
```
{"text": str}
```
See [examples/pretrain](../examples/pretrain) for a toy data and training example.

#### 2. Train

```bash
torchrun --nproc_per_node {number of gpus} \
-m retromae_pretrain.run \
--output_dir {path to save model} \
--model_name_or_path {base model} \
--train_data {path to train data} \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--max_seq_length 512
```

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). 
After training, the encoder model will saved to `{output_dir}/encoder_model`

## Fine-tune 
#### 1. Data format
Train data should be a json file, where each line is a dict like this:

```
{"query": str, "pos": List[str], "neg":List[str]}
```
See [examples/finetune](../examples/finetune) for a toy data and training example.

#### 2. Train
```
torchrun --nproc_per_node {number of gpus} \
-m finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/baai-general-embedding-large-zh \
--train_data {data file} \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--normlized True \
--temperature 0.01 \
--query_max_len 32 \
--passage_max_len 128 \
--negatives_cross_device
```
some important arguments:
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one postive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.


Other training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). 
