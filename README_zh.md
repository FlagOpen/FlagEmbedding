<h1 align="center">FlagEmbedding</h1>
<p align="center">
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made with-Python-purple">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="License" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding">
        <img alt="License" src="https://img.shields.io/badge/universal embedding-1.0-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#model-list>Model List</a> | 
        <a href=#usage>Usage</a>  |
        <a href="#evaluation">Evaluation</a> |
        <a href="#train">Train</a> |
        <a href="#contact">Contact</a> |
        <a href="#license">License</a> 
    <p>
</h4>

[English](README.md) | [中文](README_zh.md)


将任意文本映射为低维稠密向量，以用于检索、分类、聚类或语义匹配等任务，并可支持为大模型调用外部知识。

************* 🌟**Updates**🌟 *************
- 08/05/2023: 发布更小的模型(base, small), **在同尺寸模型中取得最好的性能！ 🤗**
- 08/02/2023: :tada: :tada: 发布中英文向量模型BGE(BAAI General Embedding的缩写), **在MTEB和C-MTEB榜单上取得最好的性能** 
- 08/01/2023: 发布大规模中文文本向量[评测榜单](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), 其包括31个测试任务.   




## Model List
|              Model              | Language | Description | query instruction for retrieval\* |
|:-------------------------------|:--------:| :--------:| :--------:|
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English |  :trophy: 在 [MTEB](https://huggingface.co/spaces/mteb/leaderboard) 榜单上排名**第一** | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English |  在 [MTEB](https://huggingface.co/spaces/mteb/leaderboard) 榜单上排名**第二** | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | small-scale模型，性能高于很多开源large-scale模型，推理更高效  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | :trophy: 在 [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) 榜单上排名**第一** | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) |   Chinese | 在 [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) 榜单上排名**第二** | --  |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  base-scale模型，与bge-large性能类似，但推理更快，向量维度更小 | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | small-scale模型，推理比base模型更快  | `为这个句子生成表示以用于检索相关文章：`  |

\*: 如果您需要为一个简短的查询搜索相关文档，您需要在查询中添加指令；在其他情况下，不需要指令，直接使用原始查询即可。**在任何情况下，您都不需要为候选文档增加指令**。



## Usage 

* **Using FlagEmbedding**
```
pip install -U FlagEmbedding
```
如果您使用了镜像，可能无法找到最新版的FlagEmbedding。
可以参考[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) 下载改项目进行安装。


```python
from FlagEmbedding import FlagModel
sentences = ["样例数据-1", "样例数据-2"]
model = FlagModel('BAAI/bge-large-zh', query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
embeddings_1 = model.encode(sentences)
embeddings_2 = model.encode(sentences)
smilarity = embeddings_1 @ embeddings_2.T
print(smilarity)

# 对于检索任务中的查询，请使用 encode_queries() 函数，其会自动为每个查询加上指令
# 由于候选文本不需要添加指令，检索中的候选集依然使用 encode() 或 encode_corpus() 函数
queries = ['query_1', 'query_2']
passages = ["样例段落-1", "样例段落-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```
Instruction参数 `query_instruction_for_retrieval` 请参照： [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list). 

为提高效率，FlagModel默认会使用所有的GPU进行推理。如果想要使用具体的GPU，请设置`os.environ["CUDA_VISIBLE_DEVICES"]`。


* **Sentence-Transformers**  

安装 [sentence-transformers](https://www.SBERT.net):

```
pip install -U sentence-transformers
```

基于Sentence-Transformers的使用方法:

```python
from sentence_transformers import SentenceTransformer
sentences = ["样例数据-1", "样例数据-2"]
model = SentenceTransformer('BAAI/bge-large-zh')
embeddings_1 = model.encode(sentences, normalize_embeddings=True)
embeddings_2 = model.encode(sentences, normalize_embeddings=True)
smilarity = embeddings_1 @ embeddings_2.T
print(smilarity)
```
对于检索任务，
每个查询都应该以一条指令开始(指令参考 [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). 
但对于文档，不需要添加任何指令。
``python
queries = ["手机开不了机怎么办？"]
passages = ["样例段落-1", "样例段落-2"]
instruction = "为这个句子生成表示以用于检索相关文章："
model = SentenceTransformer('BAAI/bge-large-zh')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```
* **With Langchain** 

在Langchian中使用bge模型：
```python
from langchain.embeddings import HuggingFaceInstructEmbeddings
encode_kwargs = {'normalize_embeddings': True}
model = HuggingFaceInstructEmbeddings(model_name='BAAI/bge-large-en',
                                      embed_instruction="",
                                      query_instruction="Represent this sentence for searching relevant passages: ",
                                      encode_kwargs=encode_kwargs)
```

* **HuggingFace Transformers** 

使用transformers库时，您可以这样使用模型:首先，将输入传递给transformer模型，然后选择第一个标记的最后一个隐藏状态(即[CLS])作为句子嵌入。
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
# for retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```


## Evaluation  
`baai-general-embedding` 模型在MTEB和C-MTEB排行榜上都实现了**最先进的性能**!
更多细节和评估脚本请参见 [C_MTEB](./C_MTEB). 

- **MTEB**:   

| Model Name | Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**bge-large-en**](https://huggingface.co/BAAI/bge-large-en) |  1024 | 512 | **63.98** |  **53.9** | **46.98** | 85.8 | **59.48** | 81.56 | 32.06 | **76.21** | 
| [**bge-base-en**](https://huggingface.co/BAAI/bge-base-en) |  768 | 512 |  63.36 | 53.0 | 46.32 | 85.86 | 58.7 | 81.84 | 29.27 | 75.27 | 
| [gte-large](https://huggingface.co/thenlper/gte-large) |  1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base) 	|  768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) |  1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [**bge-small-en**](https://huggingface.co/BAAI/bge-small-en) |  384 | 512 | 62.11 |  51.82 | 44.31 | 83.78 | 57.97 | 80.72 | 30.53 | 74.37 |  
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) | 768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small) | 384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | 768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	| 768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	 | 4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	| 384 | 512 	| 56.53 | 42.69 | 41.81 | 82.41 | 58.44 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	 | 384 | 512 	| 56.26 | 41.95 | 42.35 | 82.37 | 58.04 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	| 768 | 512 	| 56.00 | 41.88 | 41.1 	| 82.54 | 53.14 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	 | 768 | 512 	| 55.27 | 33.63 | 40.21 | 85.18 | 53.09 | 81.14 | 31.39 | 69.81 |



- **C-MTEB**:  

我们建立了一个中文文本嵌入的基准测试集合C-MTEB，其包括6个任务的31个数据集。
请参阅[C_MTEB](C_MTEB/README.md)获取详细介绍。

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

本节将介绍我们用于训练通用嵌入向量的方法。
训练脚本在[FlagEmbedding](./FlagEmbedding/baai_general_embedding)中。
同时，我们提供了一些示例来进行[预训练](examples/pretrain/)和[微调](examples/finetune/)。

**1. RetroMAE Pre-train**  

我们按照 [retromae](https://github.com/staoxiao/RetroMAE) 方法对模型进行预训练，
其在检索任务中表现出了良好的性能( [参考论文](https://aclanthology.org/2022.emnlp-main.35.pdf) )。
预训练是在24块A100(40G) gpu上进行的，batch大小为720。在retromae中，编码器和解码器的掩码率分别为0.3和0.5。
使用AdamW优化器，学习率为2e-5。

**Pre-training data**:
- English: 
    - [Pile](https://pile.eleuther.ai/)
    - [wikipedia](https://huggingface.co/datasets/wikipedia)
    - [msmarco](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus)
- Chinese: 
    - [wudao](https://github.com/BAAI-WuDao/Data)


**2. Finetune**  

我们使用对比学习训练模型，输入数据的格式是一个三元组' (query, positive, negative) '。
除了三元组中的负样本，我们还使用了in-batch的负样本。我们采用 [跨设备负样本共享方法](https://github.com/microsoft/MoPQ) 
在不同的gpu之间共享负样本，这会显著地**增加负样本的数量**。
我们在48块A100(40G) gpu上训练模型，batch大小为32,768。
我们使用AdamW优化器，学习率为1e-5。
对比损失的温度系数为0.01。


同时，我们在训练中为检索任务的查询添加了instruction。
对于英语，指令是`Represent this sentence for searching relevant passages: `;
对于中文，指令是`为这个句子生成表示以用于检索相关文章：`.
在评测中，针对段落检索任务的任务需要在查询中添加指令，但不需要为段落文档添加指令。


微调脚本可以在这个存储库中访问:[FlagEmbedding](./FlagEmbedding/baai_general_embedding), 你可以用它轻松地微调你的模型。

 

**Training data**:

-对于英语，我们从 [wikipedia](https://huggingface.co/datasets/wikipedia) ， [cc-net](https://github.com/facebookresearch/cc_net) 等收集了2.3亿个文本对。
-对于中文，我们从 [悟道](https://github.com/BAAI-WuDao/Data) 、[simclue](https://github.com/CLUEbenchmark/SimCLUE)等收集了1.2亿对文本。

我们计划在将来发布训练数据集。

## Schedule
- [x] Chinese Massive Text Embedding Benchmark
- [x] release baai-general-embedding models
- [x] release codes for training
- [ ] Training Datasets 
- [ ] Multilingual model
- [ ] ...

我们将不断更新向量模型和代码，希望能促进社区的发展。

## Concat
如果您有任务疑问或者建议，欢迎提交issue和PR, 
也可以发送邮件给 Shitao Xiao(stxiao@baai.ac.cn) and  Zheng Liu(liuzheng@baai.ac.cn). 


## License
FlagEmbedding基于[MIT License](LICENSE)开源协议。发布的模型权重可商用。



