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
        <img alt="License" src="https://img.shields.io/badge/universal embedding-1.1-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#model-list>模型列表</a> | 
        <a href=#常见问题>常见问题</a> | 
        <a href=#usage>使用方法</a>  |
        <a href="#evaluation">模型评估</a> |
        <a href="#train">模型训练</a> |
        <a href="#contact">Contact</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

[English](README.md) | [中文](README_zh.md)


将任意文本映射为低维稠密向量，以用于检索、分类、聚类或语义匹配等任务，并可支持为大模型调用外部知识。

************* 🌟**Updates**🌟 *************
- 10/12/2023: 发布 [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), 专为大语言模型**各种检索增强任务设计**的英文向量模型。[论文链接](https://arxiv.org/pdf/2310.07554.pdf) :fire:
- 09/15/2023: 发布 [论文](https://arxiv.org/pdf/2309.07597.pdf) 和 [数据集](https://data.baai.ac.cn/details/BAAI-MTP).
- 09/12/2023: 更新：
    - **新增重排模型**：开源交叉编码器模型bge-reranker，具有比向量模型更强大的排序能力。非常建议使用或者微调它来重新排序向量模型返回的top-k文档，提高最终结果的相关性。
    - **更新向量模型**：发布bge-*-v1.5向量模型，缓解相似度分布问题，提升无指令情况下的检索能力（但检索任务仍建议使用指令）
- 09/07/2023: 更新[微调代码](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md): 增加难负样本挖掘脚本，增加指令参数方便在微调中添加指令.
- 08/09/2023: BGE模型整合入Langchain, 可以在langchain中非常简单的[使用它](#using-langchain); C-MTEB中文榜单已[在线更新](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: 发布更小的模型(base, small), **在同尺寸模型中取得最好的性能！ 🤗**
- 08/02/2023: :tada: :tada: 发布中英文向量模型BGE(BAAI General Embedding的缩写), **在MTEB和C-MTEB榜单上取得最好的性能** 
- 08/01/2023: 发布大规模中文文本向量[评测榜单](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), 其包括31个测试任务.   




## Model List
|              Model              | Language | | Description | query instruction for retrieval [1] |
|:-------------------------------|:--------:| :--------:| :--------:|:--------:|
|  [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)  |   English | [推理](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) | 专为大语言模型各种检索增强任务设计的向量模型  | 详见 [README](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |
|  [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |   Chinese and English | [推理](#usage-for-reranker) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | 交叉编码器模型，精度比向量模型更高但推理效率较低 [2] |   |
|  [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |   Chinese and English | [推理](#usage-for-reranker) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | 交叉编码器模型，精度比向量模型更高但推理效率较低 [2] |   |
|  [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | 1.5版本，相似度分布更加合理 | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | 1.5版本，相似度分布更加合理 | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | 1.5版本，相似度分布更加合理 | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | 1.5版本，相似度分布更加合理 | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |   Chinese |  [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | 1.5版本，相似度分布更加合理 | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | 1.5版本，相似度分布更加合理 | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |  :trophy:  SOTA性能在 [MTEB](https://huggingface.co/spaces/mteb/leaderboard) 榜单 | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | base-scale 模型 | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | small-scale 模型  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: SOTA性能在 [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) 榜单 | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | base-scale 模型 | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | small-scale模型 | `为这个句子生成表示以用于检索相关文章：`  |


[1\]: 如果您需要为一个**简短的查询搜索相关的文档**，您需要在查询中添加指令；在其他情况下，不需要指令，直接使用原始查询即可。在任何情况下，您都**不需要为候选文档增加指令**。

[2\]: 不同于向量模型输出向量，reranker交叉编码器使用问题和文档作为输入，直接输出相似度。为了平衡准确率和时间成本，交叉编码器一般用于对其他简单模型检索到的top-k文档进行重排序。例如，使用bge向量模型检索前100个相关文档，然后使用bge reranker对前100个文档重新排序，得到最终的top-3结果。

## 常见问题

**1. 如何微调bge模型**

遵循这个[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) 来准备数据并微调模型。
一些建议：
- 按照这个[命令](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives) 挖掘难负样本，这可以明显提高检索性能。
- 通常，`per_device_train_batch_size`参数越大越好，其可以增大In-batch negatives的数量。可以通过开启`--fp16`, `--deepspeed df_config.json`(df_config.json请参考 [ds_config.json](./ds_config.json)), `--gradient_checkpointing`等方式来拓展batch size。
- `train_group_size` 参数在我们实验中（使用了难负样本）默认设为8。可以根据数据中的平均负样本数据进行设置：train_group_size=负样本数量+1。
- `query_max_len` 和`passage_max_len`参数应该按照实际数据长度进行设置，数据都很长的话应该增大，但不能超过512。
- 如果微调向量模型的准确率仍然不高，建议使用或者微调交叉编码器模型(bge-reranker)对top-k结果进行重新排序。
交叉编码器模型的数据格式与向量模型一致，同时也建议使用难负样本。在正确微调的情况下，交叉编码器模型的准确度会高于向量模型。
- 如果进行了预训练，预训练后的模型无法直接用于计算相似度，必须经过微调才能进行相似度。




<details>
  <summary>2. 不相似句子之间的相似度分数很高 </summary>

  <!-- ### 不相似句子之间的相似度分数很高 -->
**建议使用bge v1.5，它缓解了相似度分布的问题。** 

由于我们通过温度为0.01的对比学习来微调模型，
当前BGE模型的相似度分布大约在\[0.6, 1\]区间内。
因此，相似度大于0.6并不表示这两个句子相似。

对于下游任务，如段落检索或语义相似性，
**重要的是分数的相对顺序，而不是绝对值。**
如果你需要根据相似度阈值过滤相似句子，
请根据数据的相似度分布(如0.8,0.85，甚至0.9)选择合适的相似度阈值。

</details>


<details>
  <summary>3. 什么时候需要添加查询指令 </summary>

  <!-- ### 什么时候需要添加查询指令 -->
对于`bge-*-v1.5`，我们提高了它在不使用指令时的检索能力。无指令检索性能仅比使用指令检索性能略有下降。
因此，如果想要更加方便的话，您在所有情况下都可以在没有指令的情况下生成向量。

对于一个使用短查询寻找相关长文档的检索任务，查询与文档之间长度非常不一致，推荐为短查询添加指令。其他任务，推荐不添加指令。
**最好的选择方式，是根据实际情况选择其中表现最好的方式。**
在所有情况下，文档端都不用添加指令，只是查询端可以选择是否添加指令。

</details>





## Usage 

### Usage for Embedding Model

这里展示了一些通过
[FlagEmbedding](#using-flagembedding), [Sentence-Transformers](#using-sentence-transformers), [Langchain](#using-langchain), or [Huggingface Transformers](#using-huggingface-transformers).
使用`bge`模型的方法。

#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```
如果您使用了镜像，可能无法找到最新版的FlagEmbedding。
可以参考[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) 下载改项目进行安装。


```python
from FlagEmbedding import FlagModel
sentences = ["样例数据-1", "样例数据-2"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # 设置use_fp16为True可以加快计算，效果会稍有下降
embeddings_1 = model.encode(sentences)
embeddings_2 = model.encode(sentences)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# 对于短查询到长文档的检索任务，请对查询使用 encode_queries() 函数，其会自动为每个查询加上指令
# 由于候选文本不需要添加指令，检索中的候选集依然使用 encode() 或 encode_corpus() 函数
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```
Instruction参数 `query_instruction_for_retrieval` 请参照： [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list). 
当加载你微调后的模型时，如果你没有在训练的json文件中为query添加指令，则将其设置为空字符串`""`; 如果你在训练数据中为query添加了指令，更改为你新设置的指令。

FlagModel支持GPU也支持CPU推理。如果GPU可用，其默认优先使用GPU。如果想禁止其使用GPU，设置`os.environ["CUDA_VISIBLE_DEVICES"]=""`
为提高效率，FlagModel默认会使用所有的GPU进行推理。如果想要使用具体的GPU，请设置`os.environ["CUDA_VISIBLE_DEVICES"]`。


#### Using Sentence-Transformers

安装 [sentence-transformers](https://www.SBERT.net):

```
pip install -U sentence-transformers
```

基于Sentence-Transformers的使用方法:

```python
from sentence_transformers import SentenceTransformer
sentences = ["样例数据-1", "样例数据-2"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences, normalize_embeddings=True)
embeddings_2 = model.encode(sentences, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```
对于短查询到长文档的检索任务，
每个查询都应该以一条指令开始(指令参考 [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). 
但对于文档，不需要添加任何指令。
```python
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```
如果想使用sentence_transformers加载你微调后的模型，可以参考[这里](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding#3-load-your-model) 。
同时，对于`instruction`, 如果你没有在训练的json文件中为query添加指令，则将其设置为空字符串`""`; 如果你在训练数据中为query添加了指令，更改为你新设置的指令。


#### Using Langchain

在Langchian中使用bge模型：
```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
```

#### Using HuggingFace Transformers

使用transformers库时，您可以这样使用模型:首先，将输入传递给transformer模型，然后选择第一个标记的最后一个隐藏状态(即[CLS])作为句子嵌入。
```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# 对于短查询到长文档的检索任务, 为查询加上指令
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


### Usage for Reranker

不同于向量模型，reranker无法对单个文本输出向量，其需要输入一个文本对直接计算分数。
你可以通过在reranker中输入query和passage来获得相关度分数，分数越高代表越相关。
该重排序器基于交叉熵损失进行优化，因此相关性分数没有一个特定的数值范围。

#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```

计算相关分数，越高表示越相关:
```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) #设置 fp16 为True可以加快推理速度，效果会有可以忽略的下降

score = reranker.compute_score(['query', 'passage']) # 计算 query 和 passage的相似度
print(score)

scores = reranker.compute_score([['query 1', 'passage 1'], ['query 2', 'passage 2']])
print(scores)
```


#### Using Huggingface transformers

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```


## Evaluation  
`baai-general-embedding` 模型在MTEB和C-MTEB排行榜上都实现了**最先进的性能**!
更多细节和评估脚本请参见 [C_MTEB](./C_MTEB). 

- **MTEB**:   

| Model Name | Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**BAAI/bge-large-en-v1.5**](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1024 | 512 |  **64.23** | **54.29** |  46.08 | 87.12 | 60.03 | 83.11 | 31.61 | 75.97 |  
| [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5) |  768 | 512 | 63.55 | 53.25 |   45.77 | 86.55 | 58.86 | 82.4 | 31.07 | 75.53 |  
| [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) |  384 | 512 | 62.17 |51.68 | 43.82 |  84.92 | 58.36 | 81.59 | 30.12 | 74.14 |  
| [**bge-large-en**](https://huggingface.co/BAAI/bge-large-en) |  1024 | 512 | 63.98 |  53.9 | 46.98 | 85.8 | 59.48 | 81.56 | 32.06 | 76.21 | 
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
| [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 1024 |  **64.53** | 70.46 | 56.25 | 81.6 | 69.13 | 65.84 | 48.99 |  
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 768 |  63.13 | 69.49 | 53.72 | 79.75 | 68.07 | 65.39 | 47.53 |  
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 512 | 57.82 | 61.77 | 49.11 | 70.41 | 63.96 | 60.92 | 44.18 |   
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | 1024 | 64.20 | 71.53 | 54.98 | 78.94 | 68.32 | 65.11 | 48.39 |
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 53 | 76.77 | **68.58** | 64.91 | **50.01** |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 768 | 62.96 | 69.53 | 54.12 | 77.5 | 67.07 | 64.91 | 47.63 |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | 58.79 | 63.66 | 48.44 | 69.89 | 67.34 | 56.00 | 48.23 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 512 | 58.27 |  63.07 | 49.45 | 70.35 | 63.64 | 61.48 | 45.09 |
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 | 56.91 | 50.47 | 63.99 | 67.52 | 59.34 | 47.68 |
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 | 54.75 | 50.42 | 64.3 | 68.2 | 59.66 | 48.88 |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 768 | 55.48 | 61.63 | 46.49 | 67.07 | 65.35 | 54.35 | 40.68 |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 384 | 55.38 | 59.95 | 45.27 | 66.45 | 65.85 | 53.86 | 45.26 |
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 43.35 | 69.56 | 64.31 | 54.28 | 45.68 |
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 42.78 | 66.62 | 61 | 49.25 | 44.39 |
| [text2vec-base](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 43.41 | 67.41 | 62.19 | 49.45 | 37.66 |
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 44.97 | 70.86 | 60.66 | 49.16 | 30.02 |

所有的模型文件都已上传到huggingface上： https://huggingface.co/BAAI. 
如果你无法连接到huggingface,可以通过智源网站进行下载： https://model.baai.ac.cn/models .


- **Reranking**:
评估脚本参考 [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/).

| Model | T2Reranking | T2RerankingZh2En\* | T2RerankingEn2Zh\* | MMarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| text2vec-base-multilingual | 64.66 | 62.94 | 62.51 | 14.37 | 48.46 | 48.6 | 50.26 |  
| multilingual-e5-small | 65.62 | 60.94 | 56.41 | 29.91 | 67.26 | 66.54 | 57.78 |  
| multilingual-e5-large | 64.55 | 61.61 | 54.28 | 28.6 | 67.42 | 67.92 | 57.4 |  
| multilingual-e5-base | 64.21 | 62.13 | 54.68 | 29.5 | 66.23 | 66.98 | 57.29 |  
| m3e-base | 66.03 | 62.74 | 56.07 | 17.51 | 77.05 | 76.76 | 59.36 |  
| m3e-large | 66.13 | 62.72 | 56.1 | 16.46 | 77.76 | 78.27 | 59.57 |  
| bge-base-zh-v1.5 | 66.49 | 63.25 | 57.02 | 29.74 | 80.47 | 84.88 | 63.64 |  
| bge-large-zh-v1.5 | 65.74 | 63.39 | 57.03 | 28.74 | 83.45 | 85.44 | 63.97 |  
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | 67.28 | 63.95 | 60.45 | 35.46 | 81.26 | 84.1 | 65.42 |  
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 67.6 | 64.03 | 61.44 | 37.16 | 82.15 | 84.18 | 66.09 |  

\* : T2RerankingZh2En 是跨语言检索数据集，使用中文检索英文， T2RerankingEn2Zh是使用英文检索中文。


## Train

### BAAI Embedding 


我们使用[retromae](https://github.com/staoxiao/RetroMAE) 对模型进行预训练，再用对比学习在大规模成对数据上训练模型。
**你可以按照我们的[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) 在本地数据上微调嵌入模型。**
我们还提供了一个[预训练示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain) 。
请注意，预训练的目标是重构文本，预训练后的模型无法直接用于相似度计算，需要进行微调之后才可以用于相似度计算。
更多关于bge的训练情况请参阅[baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) ，


### BGE Reranker

交叉编码器将对查询和答案实时计算相关性分数，这比向量模型(即双编码器)更准确，但比向量模型更耗时。
因此，它可以用来对嵌入模型返回的前k个文档重新排序。
我们在多语言数据上训练了交叉编码器，数据格式与向量模型相同，因此您可以根据我们的[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) 轻松地对其进行微调。
更多细节请参考[./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/README.md)

 

## Contact
如果您有任务疑问或者建议，欢迎提交issue和PR, 
也可以发送邮件给 Shitao Xiao(stxiao@baai.ac.cn) and  Zheng Liu(liuzheng@baai.ac.cn). 


## Citation

如果您觉得我们的工作有所帮助，请考虑点个星 :star: 和引用以下论文:
```
@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlagEmbedding基于[MIT License](LICENSE)开源协议。发布的模型权重可商用。



