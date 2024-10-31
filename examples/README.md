# Examples

- [1. Introduction](#1-Introduction)
- [2. Installation](#2-Installation)
- [3. Inference](#3-Inference)
- [4. Finetune](#4-Finetune)
- [5. Evaluation](#5-Evaluation)

## 1. Introduction

In this example, we show how to **inference**, **finetune** and **evaluate** the baai-general-embedding.

## 2. Installation

* **with pip**

```shell
pip install -U FlagEmbedding
```

* **from source**

```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .
```

For development, install as editable:

```shell
pip install -e .
```

## 3. Inference

We have provided the inference code for two types of models: the **embedder** and the **reranker**. These can be loaded using `FlagAutoModel` and `FlagAutoReranker`, respectively. For more detailed instructions on their use, please refer to the documentation for the [embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder) and [reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/reranker).

### 1. Embedder

```python
from FlagEmbedding import FlagAutoModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5', 
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode_corpus(sentences_1)
embeddings_2 = model.encode_corpus(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
```

### 2. Reranker

```python
from FlagEmbedding import FlagAutoReranker
pairs = [("样例数据-1", "样例数据-3"), ("样例数据-2", "样例数据-4")]
model = FlagAutoReranker.from_finetuned('BAAI/bge-reranker-large',
                                        use_fp16=True,
                                        devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
similarity = model.compute_score(pairs, normalize=True)
print(similarity)

pairs = [("query_1", "样例文档-1"), ("query_2", "样例文档-2")]
scores = model.compute_score(pairs)
print(scores)
```

## 4. Finetune

We support fine-tuning a variety of BGE series models, including `bge-large-en-v1.5`, `bge-m3`, `bge-en-icl`, `bge-multilingual-gemma2`, `bge-reranker-v2-m3`, `bge-reranker-v2-gemma`, and `bge-reranker-v2-minicpm-layerwise`, among others. As examples, we use the basic models `bge-large-en-v1.5` and `bge-reranker-large`. For more details, please refer to the [embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder) and [reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/reranker) sections.

If you do not have the `deepspeed` and `flash-attn` packages installed, you can install them with the following commands:
```shell
pip install deepspeed
pip install flash-attn --no-build-isolation
```

### 1. Embedder

```shell
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --cache_dir ./cache/model \
    --train_data ./finetune/embedder/example_data/retrieval \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
    --output_dir ./test_encoder_only_base_bge-large-en-v1.5 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./finetune/ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div
```

### 2. Reranker

```shell
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path BAAI/bge-reranker-large \
    --cache_dir ./cache/model \
    --train_data ./finetune/reranker/example_data/normal/examples.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --output_dir ./test_encoder_only_base_bge-reranker-large \
    --overwrite_output_dir \
    --learning_rate 6e-5 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ./finetune/ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000
```

## 5. Evaluation

We support evaluations on [MTEB](https://github.com/embeddings-benchmark/mteb), [BEIR](https://github.com/beir-cellar/beir), [MSMARCO](https://microsoft.github.io/msmarco/), [MIRACL](https://github.com/project-miracl/miracl), [MLDR](https://huggingface.co/datasets/Shitao/MLDR), [MKQA](https://github.com/apple/ml-mkqa), [AIR-Bench](https://github.com/AIR-Bench/AIR-Bench), and custom datasets. Below is an example of evaluating MSMARCO passages. For more details, please refer to the [evaluation examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/evaluation).

```shell
pip install pytrec_eval
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -m FlagEmbedding.evaluation.msmarco \
    --eval_name msmarco \
    --dataset_dir ./data/msmarco \
    --dataset_names passage \
    --splits dev dl19 dl20 \
    --corpus_embd_save_dir ./data/msmarco/corpus_embd \
    --output_dir ./data/msmarco/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite True \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./data/msmarco/msmarco_eval_results.md \
    --eval_metrics ndcg_at_10 mrr_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --embedder_batch_size 512 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --reranker_batch_size 512 \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir ./cache/model
```

