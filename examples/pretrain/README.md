# Pre-train
In this example, we show how to do pre-training using retromae, 
which can improve the retrieval performance. 

## 1. Installation
* **with pip**
```
pip install -U FlagEmbedding
```

* **from source**
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .
```
For development, install as editable:
```
pip install -e .
```


## 2. Data format
Train data should be a json file, where each line is a dict like this:
```
{"text": str}
```
See [toy_pretrain_data.jsonl](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/pretrain/toy_pretrain_data.jsonl) for a toy data file.

## 3. Train

```bash
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.retromae_pretrain.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-en \
--train_data toy_pretrain_data.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 2 \
--per_device_train_batch_size {batch size; set 1 for toy data} \
--dataloader_drop_last True \
--max_seq_length 512 \
--logging_steps 10 \
--dataloader_num_workers 12
```

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). 
After training, the encoder model will saved to `{output_dir}/encoder_model`


