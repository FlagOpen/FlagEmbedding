# Finetune
In this example, we show how to finetune the baai-general-embedding with your data.

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
{"query": str, "pos": List[str], "neg":List[str]}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts.
If you have no negative texts for a query, you can random sample some from the entire corpus as the negatives.

See [toy_finetune_data.jsonl](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/toy_finetune_data.jsonl) for a toy data file.

### Hard Negatives 

Hard negatives is a widely used method to improve the quality of sentence embedding. 
You can mine hard negatives following this command:
```bash
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--use_gpu_for_searching
```

- `input_file`: json data for finetuning. This script will retrieval top-k documents for each query, 
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save json data with mined hard negatives for finetuning
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling negative from top2-top200 documents. 
- `candidate_pool`: The pool to retrieval. Default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. 
The format of this file is the same as pretrain data. If input a candidate_pool, this script will retrieve negative from this file.
- `use_gpu_for_searching`: whether use faiss-gpu to retrieve negatives.


## 3. Train
```
torchrun --nproc_per_node {number of gpus} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir {path to save model} \
--model_name_or_path BAAI/bge-large-zh-v1.5 \
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
--logging_steps 10 \
--query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章：" 
```

**some important arguments**:
- `per_device_train_batch_size`: batch size in training. In most of cases, larger batch size will bring stronger performance. You can expand it by enabling `--fp16`, `--deepspeed ./df_config.json` (df_config.json can refer to [ds_config.json](./ds_config.json)), `--gradient_checkpointing`, etc. 
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale. 
- `temperature`: It will influence the distribution of similarity scores.
- `query_max_len`: max length for query. Please set it according the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query. You also can set it `""` to add nothing to query.

More training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)


### 4. Load your model
After fine-tuning BGE model, you can load it easily in the same way as [here(with FlagModel)](https://github.com/FlagOpen/FlagEmbedding#using-flagembedding) / [(with transformers)](https://github.com/FlagOpen/FlagEmbedding#using-huggingface-transformers).

Please replace the `query_instruction_for_retrieval` with your instruction if you set a different value for hyper-parameter `--query_instruction_for_retrieval` when fine-tuning.


### 5. Evaluate model on MSMARCO
We provide [a simple script](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding/finetune/eval_msmarco.py) to evaluate the model's performance on MSMARCO, a widely used retrieval benchmark. 

First, install `faiss`, a popular approximate nearest neighbor search library:
```bash
conda install -c conda-forge faiss-gpu
```

Next, you can check the data formats for the [msmarco corpus](https://huggingface.co/datasets/namespace-Pt/msmarco-corpus) and [evaluation queries](https://huggingface.co/datasets/namespace-Pt/msmarco). 

Finally, run the following command:

```bash
python -m FlagEmbedding.baai_general_embedding.finetune.eval_msmarco \
--encoder BAAI/bge-base-en-v1.5 \
--fp16 \
--add_instruction \
--k 100
```
**some important arguments:**
- `encoder`: specify the encoder model, which can be either a model on huggingface or a local one.
- `fp16`: use half precision for inference.
- `add_instruction`: add retrieval instruction (`Represent this sentence for searching relevant passages: `).
- `k`: specify how many nearest neighbors to retrieve for each query.

The results should be similar to
```python
{
    'MRR@1': 0.2330945558739255, 
    'MRR@10': 0.35786976395142633, 
    'MRR@100': 0.3692618036917553, 
    'Recall@1': 0.22606255969436478, 
    'Recall@10': 0.6412965616045848, 
    'Recall@100': 0.9012774594078318
}
```

A brief summary of how the script works:
1. Load the model on all available GPUs through [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). 
2. Encode the corpus and offload the embeddings in `faiss` Flat index. By default, `faiss` also dumps the index on all available GPUs.
3. Encode the queries and search `100` nearest neighbors for each query.
4. Compute Recall and MRR metrics.
