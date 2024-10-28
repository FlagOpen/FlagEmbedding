# 1. Introduction

In this example, we show how to **inference**, **finetune** and **evaluation** the baai-general-embedding.

# 2. Installation

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

# 3. Inference

We have provided the inference code for two models, namely the **embedder** and the **reranker**. These can be loaded using `FlagAutoModel` and `FlagAutoReranker`, respectively. For more detailed instructions on their use, please refer to the documentation for the [embedder](https://github.com/hanhainebula/FlagEmbedding/blob/new-flagembedding-v1/examples/inference/embedder) and [reranker](https://github.com/hanhainebula/FlagEmbedding/blob/new-flagembedding-v1/examples/inference/reranker).

## 1. Embedder

```python
from FlagEmbedding import FlagAutoModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5', 
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
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

## 2. Reranker

```python
from FlagEmbedding import FlagAutoReranker
pairs = [("样例数据-1", "样例数据-3"), ("样例数据-2", "样例数据-4")]
model = FlagAutoReranker.from_finetuned('BAAI/bge-reranker-large',
                                        use_fp16=True,
                                        devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
similarity = model.compute_score(pairs, normalize=True)
print(similarity)

pairs = [("query_1", "样例文档-1"), ("query_2", "样例文档-2")]
scores = model.compute_score(pairs)
print(scores)
```

# 4. Finetune





# 5. Evaluation

We support evaluations on MTEB, BEIR, MSMARCO, MIRACL, MLDR, MKQA, and AIR-Bench. Here, we provide an example of evaluating MSMARCO passages. For more details, please refer to the [evaluation examples](https://github.com/hanhainebula/FlagEmbedding/tree/new-flagembedding-v1/examples/evaluation).