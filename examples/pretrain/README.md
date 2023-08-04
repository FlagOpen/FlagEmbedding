# Pre-train
In this example, we show how to do pre-training useing retromae, 
which can improve the retrieval performance. 

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


## Data format
Train data should be a json file, where each line is a dict like this:
```
{"text": str}
```
See [toy_pretrain_data.jsonl]() for a toy data file.

## Train

```bash
torchrun --nproc_per_node {number of gpus} \
-m retromae_pretrain.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-noinstruct \
--train_data toy_pretrain_data.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--max_seq_length 512 \
--logging_steps 1
```

some important arguments:
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one postive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.


Other training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). 


