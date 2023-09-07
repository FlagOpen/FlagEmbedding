# Finetune
In this example, we show how to finetune the baai-general-embedding with your data.

## Installation
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
 

## Data format
Train data should be a json file, where each line is a dict like this:

```
{"query": str, "pos": List[str], "neg":List[str]}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts.
If you have no negative texts for a query, you can random sample some from the entire corpus as the negatives.

See [toy_finetune_data.jsonl]() for a toy data file.

**Hard Negatives**  

Hard negatives is a widely used method to improve the quality of sentence embedding. 
You can mine hard negatives following this command:
```bash
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--query_instruction_for_retrieval "Represent this sentence for searching relevant passages: "
```

- `input_file`: json data for finetuning. This script will retrieval top-k documents for each query, 
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save json data with mined hard negatives for finetuning
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling negative from top2-top200 documents. 
- `candidate_pool`: The pool to retrieval. Default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. 
The format of this file is the same as pretrain data. If input a candidate_pool, this script will retrieve negative from this file.
- `use_gpu_for_searching`: whether use faiss-gpu to retrieve negatives.


## Train
```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-en \
--train_data ./toy_finetune_data.jsonl \
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
--query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章：" 
```

**some important arguments**:
- `per_device_train_batch_size`: batch size in training. In most of cases, larger batch size will bring stronger performance.
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale. 
- `temperature`: the similarity will be `simi = simi/temperature` before using them to compute loss. 
A higher temperature can reduce the value of similarity between texts in downstream tasks.
- `query_max_len`: max length for query. Please set it according the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query.

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


### 3. Load your model
After fine-tuning BGE model, you can load it easily in the same way as [here(with FlagModel)](https://github.com/FlagOpen/FlagEmbedding#using-flagembedding) / [(with transformers)](https://github.com/FlagOpen/FlagEmbedding#using-huggingface-transformers).

Please replace the `query_instruction_for_retrieval` with your instruction if you set a different value for hyper-parameter `--query_instruction_for_retrieval` when fine-tuning.
