# FlagEmbedding

Map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification,  clustering, or semantic search.
It also can be used in vector database for  LLMs.

**Updates** 
- 08/02/2023: Release **`baai-general-embedding-*`** Models, ranking **1st** in MTEB and C-MTEB respectively !
- 08/01/2023: We release the **Chinese Massive Text Embedding Benchmark (C-MTEB)**, consisting of 31 test dataset.   

## Model List
|              Model              | Language | Description | query instruction for retrieval |
|:-------------------------------|:--------:| :--------:| :--------:|
|  [BAAI/baai-general-embedding-large-en-instruction](https://huggingface.co/BAAI/baai-general-embedding-large-en-instruction) |   English |  rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/baai-general-embedding-large-zh-instruction](https://huggingface.co/BAAI/baai-general-embedding-large-zh-instruction) |   Chinese | rank **1st** in [C-MTEB]() bechmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/baai-general-embedding-large-zh](https://huggingface.co/BAAI/baai-general-embedding-large-zh) |   Chinese | rank **2nd** in [C-MTEB]() bechmark | --  |


## Usage 

### Sentence-Transformers

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["样例数据-1", "样例数据-2"]
model = SentenceTransformer('BAAI/baai-general-embedding-large-zh-instruction')
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings)
```


### HuggingFace Transformers
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/baai-general-embedding-large-zh-instruction')
model = AutoModel.from_pretrained('BAAI/baai-general-embedding-large-zh-instruction')
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:")
print(sentence_embeddings)
```


### Retrieval Task
For retrieval task, when you use the model whose name ends with `-instruction`
each query should start with a instruction. 
```python
from sentence_transformers import SentenceTransformer
queries = ["手机开不了机怎么办？"]
passages = ["样例段落-1", "样例段落-2"]
instruction = "为这个句子生成表示以用于检索相关文章："
model = SentenceTransformer('BAAI/baai-general-embedding-large-zh-instruction')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```


## Evaluation Results  

- **MTEB**:   

| Model Name | Model Size (GB) | Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**baai-general-embedding-large-en-instruction**](https://huggingface.co/BAAI/baai-general-embedding-large-en-instruction) | 0.67 | 1024 | 512 | 63.34 | 53.23 | 48.47 | 86.34 | 59.87 | 81.89 | 30.55 | 72.28 |   
| [gte-large](https://huggingface.co/thenlper/gte-large) | 0.67 | 1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base) 	| 0.22 | 768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1.34 | 1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) | 4.96 | 768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.44 | 768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small) | 0.07 | 384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | - | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.13 | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | 9.73 | 768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	| 0.44 | 768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	| 28.27 | 4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	| 0.13 | 384 | 512 	| 56.53 | 42.69 | 41.81 | 82.41 | 58.44 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	| 0.09 | 384 | 512 	| 56.26 | 41.95 | 42.35 | 82.37 | 58.04 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	| 0.44 | 768 | 512 	| 56.00 | 41.88 | 41.1 	| 82.54 | 53.14 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	| 0.22 | 768 | 512 	| 55.27 | 33.63 | 40.21 | 85.18 | 53.09 | 81.14 | 31.39 | 69.81 |



- **C-MTEB**:  
We create a benchmark C-MTEB for chinese text embedding which consists of  31 datasets from 6 tasks. 
More details and evaluation scripts see [evaluation](evaluation/README.md).   
 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**baai-general-embedding-large-zh-instruction**](https://huggingface.co/BAAI/baai-general-embedding-large-zh-instruction) | 1024 | **63.84** | **71.53** | **53.23** | **78.94** | 72.26 | 62.33 | 48.39 |  
| [baai-general-embedding-large-zh](https://huggingface.co/BAAI/baai-general-embedding-large-zh) | 1024 | 63.62 | 70.55 | 50.98 | 76.77 | **72.49** | **65.63** | **50.01** |   
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 |56.91 | 48.15 | 63.99 | 70.28 | 59.34 | 47.68 |  
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 |54.75 | 48.64 | 64.3 | 71.22 | 59.66 | 48.88 |  
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 40.61 | 69.56 | 67.38 | 54.28 | 45.68 |  
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 39.41 | 66.62 | 65.29 | 49.25 | 44.39 | 
| [text2vec](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 41.71 | 67.41 | 65.18 | 49.45 | 37.66 |  
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 41.98 | 70.86 | 63.42 | 49.16 | 30.02 |  
 



## Train
This section will introduce the way we used to train the general embedding. 
The training scripts are in [universal_embedding](./universal_embedding/README.md), 
and we provide some examples to do [pre-train](examples/pretrain/README.md) and [fine-tune](examples/finetune/README.md).
To data collection is to be released in the future.

**1. RetroMAE Pre-train**  
We pre-train the model following the method [retromae](https://github.com/staoxiao/RetroMAE), 
which shows promising improvement in retrieval task (see [https://aclanthology.org/2022.emnlp-main.35.pdf](https://aclanthology.org/2022.emnlp-main.35.pdf)). 
The pre-training was conducted on 24 A100(40G) GPUs with a batch size of 720. 
In retromae, the mask ratio of encoder and decoder are 0.3, 0.5 respectively.
We used the AdamW optimizer and the learning rate is 2e-5.

**Pre-training data**:
- English: 
    - [Pile](https://pile.eleuther.ai/)
    - [wikipedia](https://huggingface.co/datasets/wikipedia)
    - [msmarco](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus)
- Chinese: 
    - Subset of [wudao](https://github.com/BAAI-WuDao/Data)
    - [baidu-baike](https://baike.baidu.com/)


**2. Finetune**  
We fine-tune the model using a contrastive objective. 
The format of input data is a triple`(query, positive, negative)`. 
Besides the negative in the triple, we also adopt in-batch negatives strategy. 

We trained our model on 48 A100(40G) GPUs with a large batch size of 32,768. 
We used the AdamW optimizer and the learning rate is 1e-5.
The sequence length was limited to 128 tokens.  
The temperature for contrastive loss is 0.01.

The finetune script is accessible in this repository: [universal_embedding](./universal_embedding/README.md). 
You can easily finetune your model with it.

**Training data**:

English:

| Dataset       | Pairs  | Number of training pairs  |
|------------|:------------:|:-------------:|
| [sentence-transformers Data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data) | -- | 119,837,845 |
| [wikipedia](https://huggingface.co/datasets/wikipedia) | (title + section title, passage) | 15,000,000  |
| [cc-net](https://github.com/facebookresearch/cc_net) | (title, passage) | 18,353,185  |
| [stackexchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl) | (title, upvoted answer) (title+body, upvoted answer)  | 10,896,142 |
| [reddit](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) | (title, body) | 56,682,431 |
| [S2orc](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) | (title, abstract) (title,  citation title) | 12,174,793 |
| Total | --  | 232,944,396  |


Chinese:

| Dataset       | Pairs  | Number of training pairs  |
|------------|:------------:|:-------------:|
| [wudao](https://github.com/BAAI-WuDao/Data) | (title, passage) | 47,099,374  |
| [cmrc2018](https://huggingface.co/datasets/cmrc2018) | (query, context)  | 10,142   |
| [dureader_robust](https://github.com/baidu/DuReader) | (query, context) |  14,520 |
| [dureader_retriever](https://github.com/baidu/DuReader) | (query, passage) |  86,395 |
| [dureader_checklist](https://github.com/baidu/DuReader) | (query, passage) |  1,404 |
| [simclue](https://github.com/CLUEbenchmark/SimCLUE) | (sentence_a, sentence_b) | 389,370 |
| [csl](https://arxiv.org/abs/2209.05034)  |  (title, abstract)  | 395,927|
| [amazon_reviews_multi](amazon_reviews_multi) | (title, body) | 200,000  |
| [wiki_atomic_edits](https://huggingface.co/datasets/wiki_atomic_edits) | (base_sentence, edited_sentence)  | 1,213,780  |
| [mlqa](https://huggingface.co/datasets/mlqa) | (question, context) | 76,285  |
| [xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum) | (title, summary) (title, text) | 93,404  |
| zhihu* | (query, answer) | 30,000,000 |
| baike* | (title, passage) | 18,257,816  |
| news*  | (title, passage)  | 5,061,912 |
| paper*  | (title, abstract)  |  948,795 | 
|   WebQA*  |  (query, answer)   |  16,609,624   |
| Total | --  | 120,458,748 |

`*`: unpublished dataset collected by BAAI.

After training on large-scale corpus, we finetune the model with 
a small-scale but high-quality supervised dataset. The sequence length was limited to 512 tokens to improve the similarity between long texts.  
- English: [Msmarco-passage](https://microsoft.github.io/msmarco/Datasets.html), [NQ](https://ai.google.com/research/NaturalQuestions), [NLI](https://github.com/princeton-nlp/SimCSE/blob/main/data/download_nli.sh), [S2orc(cited abstracts)](), [Quora](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)
- Chinese: [T2ranking](https://huggingface.co/datasets/THUIR/T2Ranking), [duretriever](https://github.com/baidu/DuReader), [nli-zh](https://huggingface.co/datasets/shibing624/nli_zh)

For the version with `*-instrcution`, we add instruction to the query for retrieval task in the training. 
For english, the instruction is `Represent this sentence for searching relevant passages: `;
For chinese, the instruction is `为这个句子生成表示以用于检索相关文章：`.
In the evaluation, the instruction should be added for sentence to passages retrieval task, not be added for other tasks.



## Citing & Authors

<!--- Describe where people can find more information -->
