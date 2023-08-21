<h1 align="center">FlagEmbedding</h1>
<p align="center">
    <a href="https://github.com/FlagOpen/FlagEmbedding">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="Build" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding">
        <img alt="Build" src="https://img.shields.io/badge/FlagEmbedding-1.0-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#model-list>Model List</a> | 
        <a href=#frequently-asked-questions>FAQ</a> |
        <a href=#usage>Usage</a>  |
        <a href="#evaluation">Evaluation</a> |
        <a href="#train">Train</a> |
        <a href="#contact">Contact</a> |
        <a href="#license">License</a> 
    <p>
</h4>


[English](README.md) | [中文](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

FlagEmbedding can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification,  clustering, or semantic search.
And it also can be used in vector database for LLMs.

************* 🌟**Updates**🌟 *************
- 08/09/2023: BGE Models are integrated into **Langchain**, you can use it like [this](#using-langchain); C-MTEB **leaderboard** is [avaliable](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: Release base-scale and small-scale models, **best performance among the models of the same size 🤗**
- 08/02/2023: Release `bge-large-*`(short for BAAI General Embedding) Models, **rank 1st on MTEB and C-MTEB benchmark!** :tada: :tada: 
- 08/01/2023: We release the [Chinese Massive Text Embedding Benchmark](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), consisting of 31 test dataset.   


## Model List

`bge` is short for `BAAI general embedding`.

|              Model              | Language | Description | query instruction for retrieval\* |
|:-------------------------------|:--------:| :--------:| :--------:|
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English |  :trophy: rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English |  rank **2nd** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | a small-scale model but with competitive performance  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | :trophy: rank **1st** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) |   Chinese | This model is trained without instruction, and rank **2nd** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark |   |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  a base-scale model but has similar ability with `bge-large-zh` | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | a small-scale model but with competitive performance | `为这个句子生成表示以用于检索相关文章：`  |

\*: If you need to search the **long** relevant passages to a **short** query (s2p retrieval task), you need to add the instruction to the query; in other cases, no instruction is needed, just use the original query directly. In all cases, **no instruction** need to be added to passages.

## Frequently asked questions

1. The similarity score between two dissimilar sentence is higher than 0.5

Since we finetune the models by contrastive learning with a temperature of 0.01, 
the similarity distribution of the current BGE model is about in the interval \[0.6, 1\].
So a similarity score greater than 0.5 does not indicate that the two sentence are similar.

For downstream tasks, such as passage retrieval or semantic similarity, 
**what matters is the relative order of the scores, not the absolute value.**
If you need to filter similar sentences based on a similarity threshold, 
please select an appropriate similarity threshold based on the similarity distribution on your data (such as 0.8, 0.85, or even 0.9).
If you want to adjust the similarity distribution, you can fine-tune the models on your data with a higher temperature.

2. When do the query instruction need to be used

For a retrieval task that uses short queries to find long related documents, 
it is recommended to add instructions for these short queries.
For other tasks, it is recommended not to add instructions. 
For example, in Quora task, which needs to use a short question to search another related short questions, 
the instruction is not recommended to add. 
The best method to decide whether to add instructions for queries is choosing the setting which can achieve better performance in your task.
In all cases, the documents/passages do not need to add the instruction, only need to consider whether to add the instruction for queries.

 


## Usage 

Here are some examples to use `bge` models with 
[FlagEmbedding](#using-flagembedding), [Sentence-Transformers](#using-sentence-transformers), [Langchain](#using-langchain), or [Huggingface Transformers](#using-huggingface-transformers).

#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```
If it doesn't work for you, you can see [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) for more methods to install FlagEmbedding.

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh', query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, please use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```
The value of argument `query_instruction_for_retrieval` see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list). 
To load a your fine-tuned model, use your instruction if you add it during fine-tuning. Set it a empty string "" if you don't add instruction to query in your json file.

FlagModel will use all available GPUs when encoding, please set `os.environ["CUDA_VISIBLE_DEVICES"]` to choose GPU.
You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make GPUs unavailable.


#### Using Sentence-Transformers

Using this model also is easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```
For s2p(short query to long passage) retrieval task, 
each short query should start with an instruction (instructions see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). 
But the instruction is not needed for passages.
```python
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```
If you want to load a your fine-tuned models, see [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding#3-load-your-model). 
And use your instruction if you add it during fine-tuning. Set it a empty string "" if you don't add instruction to query in your json file.

#### Using Langchain 

You can use `bge` in langchain like this:
```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
query_instruction="为这个句子生成表示以用于检索相关文章："
)
```


#### Using HuggingFace Transformers

With transformers package, you can use the model like this: First, you pass your input through the transformer model, then you select the last hidden state of first token (i.e., [CLS]) as the sentence embedding.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh')
model = AutoModel.from_pretrained('BAAI/bge-large-zh')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```


## Evaluation  
`baai-general-embedding` models achieve **state-of-the-art performance on both MTEB and C-MTEB leaderboard!**
More details and evaluation tools see our [scripts](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md). 

- **MTEB**:   

| Model Name |  Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**bge-large-en**](https://huggingface.co/BAAI/bge-large-en) |  1024 | 512 | **63.98** |  **53.9** | **46.98** | 85.8 | **59.48** | 81.56 | 32.06 | **76.21** | 
| [**bge-base-en**](https://huggingface.co/BAAI/bge-base-en) |  768 | 512 |  63.36 | 53.0 | 46.32 | 85.86 | 58.7 | 81.84 | 29.27 | 75.27 | 
| [gte-large](https://huggingface.co/thenlper/gte-large) |  1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base) 	|  768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) |  1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [**bge-small-en**](https://huggingface.co/BAAI/bge-small-en) |  384 | 512 | 62.11 |  51.82 | 44.31 | 83.78 | 57.97 | 80.72 | 30.53 | 74.37 |  
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) |  768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) |  768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small) |  384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) |  768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	|  768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	|  4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	|  384 | 512 	| 56.53 | 42.69 | 41.81 | 82.41 | 58.44 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	|  384 | 512 	| 56.26 | 41.95 | 42.35 | 82.37 | 58.04 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	|  768 | 512 	| 56.00 | 41.88 | 41.1 	| 82.54 | 53.14 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	|  768 | 512 	| 55.27 | 33.63 | 40.21 | 85.18 | 53.09 | 81.14 | 31.39 | 69.81 |



- **C-MTEB**:  
We create a benchmark C-MTEB for chinese text embedding which consists of  31 datasets from 6 tasks. 
Please refer to [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md) for a detailed introduction.
 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 1024 | **64.20** | **71.53** | **53.23** | **78.94** | 72.26 | **65.11** | 48.39 |  
| [**bge-large-zh-noinstruct**](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 50.98 | 76.77 | **72.49** | 64.91 | **50.01** |   
| [**BAAI/bge-base-zh**](https://huggingface.co/BAAI/bge-base-zh) |  768 | 62.96 | 69.53 | 52.05 | 77.5 | 70.98 | 64.91 | 47.63 |  
| [**BAAI/bge-small-zh**](https://huggingface.co/BAAI/bge-small-zh) | 512 | 58.27 |  63.07 | 46.87 | 70.35 | 67.78 | 61.48 | 45.09 |  
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 |56.91 | 48.15 | 63.99 | 70.28 | 59.34 | 47.68 |  
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 |54.75 | 48.64 | 64.3 | 71.22 | 59.66 | 48.88 |  
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 40.61 | 69.56 | 67.38 | 54.28 | 45.68 |  
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 39.41 | 66.62 | 65.29 | 49.25 | 44.39 | 
| [text2vec](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 41.71 | 67.41 | 65.18 | 49.45 | 37.66 |  
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 41.98 | 70.86 | 63.42 | 49.16 | 30.02 |  



## Train
This section will introduce the way we used to train the general embedding. 
The training scripts are in [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md), 
and we provide some examples to do [pre-train](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/pretrain/README.md) and [fine-tune](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/README.md).


**1. RetroMAE Pre-train**  
We pre-train the model following the method [retromae](https://github.com/staoxiao/RetroMAE), 
which shows promising improvement in retrieval task ([paper](https://aclanthology.org/2022.emnlp-main.35.pdf)). 
The pre-training was conducted on 24 A100(40G) GPUs with a batch size of 720. 
In retromae, the mask ratio of encoder and decoder are 0.3, 0.5 respectively.
We used the AdamW optimizer and the learning rate is 2e-5.

**Pre-training data**:
- English: 
    - [Pile](https://pile.eleuther.ai/)
    - [wikipedia](https://huggingface.co/datasets/wikipedia)
    - [msmarco](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus)
- Chinese: 
    - [wudao](https://github.com/BAAI-WuDao/Data)


**2. Finetune**  
We fine-tune the model using a contrastive objective. 
The format of input data is a triple`(query, positive, negative)`. 
Besides the negative in the triple, we also adopt in-batch negatives strategy. 
We employ the cross-device negatives sharing method to share negatives among different GPUs, 
which can dramatically **increase the number of negatives**.

We trained our model on 48 A100(40G) GPUs with a large batch size of 32,784 (so there are **65,567** negatives for each query in a batch). 
We used the AdamW optimizer and the learning rate is 1e-5.
The temperature for contrastive loss is 0.01.

Besides, we add instruction to the query for s2p(short query to long passage) retrieval task in the training (add nothing to passages). 
For English, the instruction is `Represent this sentence for searching relevant passages: `;
For Chinese, the instruction is `为这个句子生成表示以用于检索相关文章：`.
In the evaluation, the instruction should be added for queries in retrieval task, not be added for other tasks.
Noted that the instruction is not needed for passages.

The finetune script is accessible in this repository: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md). 
You can easily finetune your model with it.

**Training data**:

- For English, we collect 230M text pairs from [wikipedia](https://huggingface.co/datasets/wikipedia), [cc-net](https://github.com/facebookresearch/cc_net), and so on.

- For chinese, we collect 120M text pairs from [wudao](https://github.com/BAAI-WuDao/Data), [simclue](https://github.com/CLUEbenchmark/SimCLUE), and so on.

**The data collection is to be released in the future.**


## Schedule
- [x] Chinese Massive Text Embedding Benchmark
- [x] release baai-general-embedding models
- [x] release codes for training
- [ ] Multilingual model
- [ ] Training Datasets 
- [ ] ...

We will continually update the embedding models and training codes, 
hoping to promote the development of the embedding model community.


## Contact
If you have any question or suggestion related to this project, feel free to open an issue or pull a request.
You also can email Shitao Xiao(stxiao@baai.ac.cn) and Zheng Liu(liuzheng@baai.ac.cn). 


## License
FlagEmbedding is licensed under [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). The released models can be used for commercial purposes free of charge.



