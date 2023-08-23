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


## Train
```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-en \
--train_data toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.01 \
--query_max_len 32 \
--passage_max_len 128 \
--train_group_size 2 \
--negatives_cross_device 
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

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


### 3. Load your model
After fine-tuning BGE model, you can load it easily in the same way as [here(with FlagModel)](https://github.com/FlagOpen/FlagEmbedding#using-flagembedding) / [(with transformers)](https://github.com/FlagOpen/FlagEmbedding#using-huggingface-transformers).

Please replace the `query_instruction_for_retrieval` with your instruction if you add a instruction for query in your data json.

If you don't add instruction for query in your data, please set `query_instruction_for_retrieval` to be a `""`.

```python
from FlagEmbedding import FlagModel
model = FlagModel('BAAI/bge-large-zh', query_instruction_for_retrieval="")

queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```

If you want to load your fine-tuned models with `sentence_transformers`, you should **set the pooling_mode to be `cls`** (the default pooling method in sentence_transformers is mean pooling).
You can load your model like this:
```python
from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer(finetuned_model_path, max_seq_length=512, do_lower_case=True)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

