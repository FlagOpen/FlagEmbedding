# Embedding Model

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

**we provide some examples to [pre-train](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/pretrain/README.md) and [fine-tune](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/README.md) your model.**


## Training process

This section will introduce the way we used to train the general embedding. 


### 1. Pre-train
We pre-train the model following the method [RetroMAE](https://github.com/staoxiao/RetroMAE), 
which shows promising improvements in retrieval tasks. 
The pre-training was conducted on 24 A100(40G) GPUs with a batch size of 720. 
In RetroMAE, the mask ratio of the encoder and decoder are 0.3 and 0.5, respectively.

**You can pretrain your model following our [example](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/pretrain/README.md).**


**Pre-training data**:
- English: [Pile](https://pile.eleuther.ai/), [wikipedia](https://huggingface.co/datasets/wikipedia), and [msmarco](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus)
- Chinese: [wudao](https://data.baai.ac.cn/details/WuDaoCorporaText)


### 2. Train 

We fine-tune the model using a contrastive objective. 
The format of input data is a triple `(query, positive, negative)`. 
Besides the negative in the triple, we also adopt in-batch negatives strategy. 
We employ the cross-device negatives sharing method to share negatives among different GPUs, 
which can dramatically increase the number of negatives. We used the AdamW optimizer and the learning rate is 1e-5/2e-5/3e-5 for large/base/small scale.
The temperature is 0.01 for `bge`, and 0.02 for `bge-v1.5`.

The fine-tune script is accessible in this repository: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md). 
**You can easily fine-tune your model following our [example](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/README.md).**

To get a general embedding model, the training of our embedding model consists of two stages: 
Firstly, we train embedding model on a massive-scale pair data from unsupervised data, 
and then optimize it with limited labeled data.

#### 2.1 unsupervised pairs

We mine a large-scale pairs data from various domains. 
And we use a simple strategy to filter the data. Particularly,
we use a third party model: [text2vec](https://huggingface.co/shibing624/text2vec-base-chinese)
to score the strength of relation for each text pair.
We empirically choose a threshold 0.43, and drop
the samples whose scores are below the threshold.
Using automatic mixed precision training and gradient checkpointing method,.
For english, we trained our model on 48 A100(40G) GPUs with a large batch size of 32,784. 
For chinese, we trained our model on 24 A100(40G) GPUs with a large batch size of 19,200.  


**Training data**:

- English: [sentence-transformers Data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), 
[wikipedia](https://huggingface.co/datasets/wikipedia), [cc-net](https://github.com/facebookresearch/cc_net), [stackexchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl),
[reddit](https://huggingface.co/datasets/sentence-transformers/reddit-title-body), [S2orc](https://huggingface.co/datasets/sentence-transformers/reddit-title-body)

- Chinese: [wudao](https://data.baai.ac.cn/details/WuDaoCorporaText), [cmrc2018](https://huggingface.co/datasets/cmrc2018), [dureader](https://github.com/baidu/DuReader),
[simclue](https://github.com/CLUEbenchmark/SimCLUE), [csl](https://arxiv.org/abs/2209.05034), [amazon_reviews_multi](amazon_reviews_multi),
[wiki_atomic_edits](https://huggingface.co/datasets/wiki_atomic_edits),  [mlqa](https://huggingface.co/datasets/mlqa),
[xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum), and other data collected by BAAI teams from internet (including QA, news and paper).

We release the dataset at https://data.baai.ac.cn/details/BAAI-MTP .

#### 2.2 high-quality supervised pairs

In this stage, we train the embedding model using the supervised data from multiple tasks. 
For each task, we mine hard negatives using the model trained on unsupervised pairs. 
The script to mind hard negatives is [hn_mine.py](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/finetune/hn_mine.py), and you can use it following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#data-format).

Besides, we add an instruction to the query for retrieval tasks in the training (add nothing to passages). 
For English, the instruction is `Represent this sentence for searching relevant passages: `;
For Chinese, the instruction is `为这个句子生成表示以用于检索相关文章：`.
At evaluation, the instruction is recommend to be added for queries in the retrieval task, but not be added for other tasks.
Note that the instruction is not needed for passages.


**Training data**:
- English: 
    - data from [sentence-transformers](https://huggingface.co/datasets/sentence-transformers/embedding-training-data): msmarco, nq, hotpotqa, quora, stackexchange_duplicate, s2orc
    - Natural Language interface: [NLI](https://github.com/princeton-nlp/SimCSE)
    - others: [MEDI](https://github.com/xlang-ai/instructor-embedding), [fever](https://github.com/awslabs/fever)

- Chinese
    - Web Search: [T2ranking](https://huggingface.co/datasets/THUIR/T2Ranking), [MMmarco](https://github.com/unicamp-dl/mMARCO)
    - QA: [dulreader](https://github.com/baidu/DuReader)
    - Natural Language interface: [nli-zh](https://huggingface.co/datasets/shibing624/nli_zh)







