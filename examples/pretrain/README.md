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
--model_name_or_path BAAI/baai-general-embedding-large-zh \
--train_data toy_pretrain_data.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--logging_steps 1
```
More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
After training, the encoder model will saved to `{output_dir}/encoder_model`


