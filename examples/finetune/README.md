# Finetune
In this example, we show how to finetune the baai-general-embedding with a your data.

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
{"query": str, "pos": List[str], "neg":List[str]}
```
See [toy_finetune_data.jsonl]() for a toy data file.


## Train
```
python -m torch.distributed.launch --nproc_per_node {number of gpus} \
-m finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/baai-general-embedding-large-zh-instruction \
--train_data toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--normlized True \
--temperature 0.01 \
--per_device_train_batch_size  1 
```

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)





