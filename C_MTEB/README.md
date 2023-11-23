<h1 align="center">Chinese Massive Text Embedding Benchmark</h1>
<p align="center">
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="Build" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Made with-Python-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#installation>Installation</a> | 
        <a href=#evaluation>Evaluation</a>  |
        <a href="#leaderboard">Leaderboard</a> |
        <a href="#tasks">Tasks</a> |
        <a href="#acknowledgement">Acknowledgement</a> |
    <p>
</h4>


## Installation
C-MTEB is devloped based on [MTEB](https://github.com/embeddings-benchmark/mteb). 
```
pip install -U C_MTEB
```
Or clone this repo and install as editable
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding/C_MTEB
pip install -e .
```

## Evaluation

### Evaluate reranker
```bash
python eval_cross_encoder.py --model_name_or_path BAAI/bge-reranker-base
```
 
### Evaluate embedding model
* **With our scripts**

You can **reproduce the results of `baai-general-embedding (bge)`** using the provided python script (see [eval_C-MTEB.py](./eval_C-MTEB.py) )
```bash
python eval_C-MTEB.py --model_name_or_path BAAI/bge-large-zh

# for MTEB leaderboard
python eval_MTEB.py --model_name_or_path BAAI/bge-large-en

```

* **With sentence-transformers** 
 
You can use C-MTEB easily in the same way as [MTEB](https://github.com/embeddings-benchmark/mteb).

Note that the original sentence-transformers model doesn't support instruction. 
So this method cannot test the performance of `bge-*` models.

```python
from mteb import MTEB
from C_MTEB import *
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "bert-base-uncased"

model = SentenceTransformer(model_name)
evaluation = MTEB(task_langs=['zh'])
results = evaluation.run(model, output_folder=f"zh_results/{model_name}")
```


* **Using a custom model**  
To evaluate a new model, you can load it via sentence_transformers if it is supported by sentence_transformers.
Otherwise, models should be implemented like below (implementing an `encode` function taking as input a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.).): 

```python
class MyModel():
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        pass

model = MyModel()
evaluation = MTEB(tasks=["T2Retrival"])
evaluation.run(model)
```


## Leaderboard

### 1. Reranker

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

\* : T2RerankingZh2En and T2RerankingEn2Zh are cross-language retrieval task


### 2. Embedding 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 1024 |  **64.53** | 70.46 | 56.25 | 81.6 | 69.13 | 65.84 | 48.99 |  
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 768 |  63.13 | 69.49 | 53.72 | 79.75 | 68.07 | 65.39 | 47.53 |  
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 512 | 57.82 | 61.77 | 49.11 | 70.41 | 63.96 | 60.92 | 44.18 |   
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | 1024 | 64.20 | 71.53 | 54.98 | 78.94 | 68.32 | 65.11 | 48.39 |
| [BAAI/bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 53 | 76.77 | 68.58 | 64.91 | 50.01 |
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


### 2.1. Retrieval
| Model | T2Retrieval | MMarcoRetrieval | DuRetrieval | CovidRetrieval | CmedqaRetrieval | EcomRetrieval | MedicalRetrieval | VideoRetrieval | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 58.67 | 55.31 | 59.36 | 55.48 | 18.04 | 40.48 | 29.8 | 38.04 | 44.4 |  
| text2vec-large-chinese | 50.52 | 45.96 | 51.87 | 60.48 | 15.53 | 37.58 | 30.93 | 42.65 | 41.94 |  
| text2vec-base-chinese | 51.67 | 44.06 | 52.23 | 44.81 | 15.91 | 34.59 | 27.56 | 39.52 | 38.79 |  
| m3e-base | 73.14 | 65.45 | 75.76 | 66.42 | 30.33 | 50.27 | 42.8 | 51.11 | 56.91 |  
| m3e-large | 72.36 | 61.06 | 74.69 | 61.33 | 30.73 | 45.18 | 48.66 | 44.02 | 54.75 |  
| OpenAI(text-embedding-ada-002) | 69.14 | 69.86 | 71.17 | 57.21 | 22.36 | 44.49 | 37.92 | 43.85 | 52.0 |  
| multilingual-e5-small | 71.39 | 73.17 | 81.35 | 72.82 | 24.38 | 53.56 | 44.84 | 58.09 | 59.95 |
| multilingual-e5-base | 70.86 | 76.04 | 81.64 | 73.45 | 27.2 | 54.17 | 48.35 | 61.3 | 61.63 |
| multilingual-e5-large | 76.11 | 79.2 | 85.32 | 75.51 | 28.67 | 54.75 | 51.44 | 58.25 | 63.66 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 77.59 | 67.56 | 77.89 | 68.95 | 35.18 | 58.17 | 49.9 | 69.33 | 63.07 |  
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 83.35 | 79.11 | 86.02 | 72.07 | 41.77 | 63.53 | 56.64 | 73.76 | 69.53 |  
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 84.39 | 81.38 | 84.68 | 75.07 | 41.03 | 65.6 | 58.28 | 73.94 | 70.55 |  
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 84.82 | 81.28 | 86.94 | 74.06 | 42.4 | 66.12 | 59.39 | 77.19 | 71.53 |  


### 2.2.  STS  
| Model | ATEC | BQ | LCQMC | PAWSX | STSB | AFQMC | QBQTC | STS22 (zh) | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| luotuo-bert-medium | 30.84 | 43.33 | 66.74 | 12.31 | 73.22 | 22.24 | 27.2 | 66.4 | 42.78 |
| text2vec-large-chinese | 32.45 | 44.22 | 69.16 | 14.55 | 79.45 | 24.51 | 29.51 | 65.94 | 44.97 |
| text2vec-base-chinese | 31.93 | 42.67 | 70.16 | 17.21 | 79.3 | 26.06 | 24.62 | 55.35 | 43.41 |
| m3e-base | 41.27 | 63.81 | 74.88 | 12.19 | 76.97 | 35.87 | 32.07 | 66.73 | 50.47 |
| m3e-large | 41.8 | 65.2 | 74.2 | 15.95 | 74.16 | 36.53 | 32.65 | 62.91 | 50.42 |
| OpenAI(text-embedding-ada-002) | 29.25 | 45.33 | 68.41 | 16.55 | 70.61 | 23.88 | 30.27 | 62.53 | 43.35 |
| multilingual-e5-small | 35.14 | 43.27 | 72.7 | 11.01 | 77.73 | 25.21 | 30.25 | 66.84 | 45.27 |
| multilingual-e5-base | 37.01 | 45.45 | 74.15 | 12.14 | 79.05 | 29.67 | 28.81 | 65.64 | 46.49 |
| multilingual-e5-large | 39.81 | 46.44 | 75.95 | 14.63 | 81.08 | 33.02 | 29.77 | 66.82 | 48.44 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 43.17 | 55.47 | 72.61 | 9.97 | 76.48 | 33.93 | 36.45 | 67.54 | 49.45 |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 48.28 | 61.21 | 74.98 | 20.65 | 78.66 | 42.53 | 38.01 | 68.64 | 54.12 |
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 48.29 | 60.53 | 74.71 | 16.64 | 78.41 | 43.06 | 35.2 | 67.19 | 53 |
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 49.75 | 62.93 | 75.45 | 22.45 | 78.51 | 44.57 | 38.92 | 67.24 | 54.98 |


### 2.3. PairClassification  
| Model | Ocnli | Cmnli | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 60.7 | 72.55 | 66.62 |  
| text2vec-large-chinese | 64.04 | 77.67 | 70.86 |  
| text2vec-base-chinese | 60.95 | 73.87 | 67.41 |  
| m3e-base | 58.0 | 69.98 | 63.99 |  
| m3e-large | 59.33 | 69.27 | 64.3 |  
| OpenAI(text-embedding-ada-002) | 63.08 | 76.03 | 69.56 |
| multilingual-e5-small | 60.77 | 72.12 | 66.45 |
| multilingual-e5-base | 59.63 | 74.51 | 67.07 |
| multilingual-e5-large | 78.18 | 78.18 | 69.89 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 65.25 | 75.46 | 70.35 |  
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 73.32 | 81.69 | 77.5 |  
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 71.37 | 82.17 | 76.77 |  
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 75.75 | 82.12 | 78.94 |  


### 2.4. Classification  
| Model | TNews | IFlyTek | MultilingualSentiment | JDReview | OnlineShopping | Waimai | AmazonReviewsClassification (zh) | MassiveIntentClassification (zh-CN) | MassiveScenarioClassification (zh-CN) | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 45.22 | 41.75 | 61.21 | 79.68 | 84.3 | 79.57 | 34.46 | 57.47 | 65.32 | 61 |
| text2vec-large-chinese | 38.92 | 41.54 | 58.97 | 81.56 | 83.51 | 76.01 | 33.77 | 63.23 | 68.45 | 60.66 |
| text2vec-base-chinese | 43.02 | 42.05 | 60.98 | 82.14 | 85.69 | 77.22 | 34.12 | 63.98 | 70.52 | 62.19 |
| m3e-base | 48.28 | 44.42 | 71.9 | 85.33 | 87.77 | 83.99 | 43.02 | 68.4 | 74.6 | 67.52 |
| m3e-large | 48.26 | 43.96 | 72.47 | 86.92 | 89.59 | 86.1 | 44.44 | 67.23 | 74.88 | 68.2 |
| OpenAI(text-embedding-ada-002) | 45.77 | 44.62 | 67.99 | 74.6 | 88.94 | 82.37 | 38.3 | 64.81 | 71.4 | 64.31 |
| multilingual-e5-small | 48.38 | 47.35 | 64.74 | 79.34 | 88.73 | 83.9 | 37.5 | 68.24 | 74.47 | 65.85 |
| multilingual-e5-base | 47.06 | 44.93 | 65.28 | 76.21 | 88.4 | 84.42 | 37.23 | 69.16 | 75.42 | 65.35 |
| multilingual-e5-large | 48.38 | 45.47 | 68.58 | 80.99 | 90.81 | 85.02 | 38.83 | 71.12 | 76.83 | 67.34 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 47.67 | 42.07 | 65.07 | 80.64 | 87.4 | 83.8 | 37.31 | 61.44 | 67.39 | 63.64 |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 49.97 | 44.54 | 70.63 | 83.92 | 91.38 | 85.46 | 40.68 | 65.72 | 71.3 | 67.07 | 
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 52.05 | 45.32 | 73.7 | 85.38 | 91.66 | 86.83 | 41.94 | 66.96 | 73.39 | 68.58 |
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 50.84 | 45.09 | 74.41 | 85.08 | 91.6 | 86.54 | 42.39 | 67.18 | 71.76 | 68.32 | 


### 2.5. Reranking  
| Model | T2Reranking | MmarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 65.76 | 14.55 | 57.82 | 58.88 | 49.25 |  
| text2vec-large-chinese | 64.82 | 12.48 | 58.92 | 60.41 | 49.16 |  
| text2vec-base-chinese | 65.95 | 12.76 | 59.26 | 59.82 | 49.45 | 
| m3e-base | 66.03 | 17.51 | 77.05 | 76.76 | 59.34 |   
| m3e-large | 66.13 | 16.46 | 77.76 | 78.27 | 59.66 |  
| OpenAI(text-embedding-ada-002) | 66.65 | 23.39 | 63.08 | 64.02 | 54.28 |
| multilingual-e5-small | 65.24 | 24.33 | 63.44 | 62.41 | 53.86 |
| multilingual-e5-base | 64.39 | 21.76 | 65.21 | 66.06 | 54.35 |
| multilingual-e5-large | 65.83 | 21.34 | 68.25 | 68.56 | 56.00 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 66.2 | 22.82 | 77.08 | 79.82 | 61.48 |  
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 66.49 | 28.24 | 80.12 | 84.78 | 64.91 |  
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 66.16 | 27.1 | 81.72 | 84.64 | 64.91 |  
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 66.19 | 26.23 | 83.01 | 85.01 | 65.11 |  

### 2.6. Clustering  
| Model | CLSClusteringS2S | CLSClusteringP2P | ThuNewsClusteringS2S | ThuNewsClusteringP2P | Avg |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| luotuo-bert-medium | 33.46 | 37.01 | 48.26 | 58.83 | 44.39 |
| text2vec-large-chinese | 28.77 | 30.13 | 26.14 | 35.05 | 30.02 |
| text2vec-base-chinese | 32.42 | 35.27 | 40.01 | 42.92 | 37.66 |
| m3e-base | 37.34 | 39.81 | 53.78 | 59.77 | 47.68 |
| m3e-large | 38.02 | 38.6 | 58.51 | 60.39 | 48.88 |
| OpenAI(text-embedding-ada-002) | 35.91 | 38.26 | 49.86 | 58.71 | 45.68 |
| multilingual-e5-small | 37.79 | 39.14 | 48.93 | 55.18 | 45.26 |
| multilingual-e5-base | 36.99 | 32.41 | 52.36 | 40.98 | 40.68 |
| multilingual-e5-large | 38.59 | 40.68 | 55.59 | 58.05 | 48.23 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 34.34 | 38.23 | 51.84 | 55.95 | 45.09 |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 36.59 | 38.79 | 56.16 | 59.0 | 47.63 |
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 40.04 | 41.23 | 56.75 | 62.03 | 50.01 |
| [**bge-large-zh**](https://huggingface.co/BAAI/bge-large-zh) | 38.05 | 40.92 | 58.79 | 55.79 | 48.39 |



## Tasks

An overview of tasks and datasets available in MTEB-chinese is provided in the following table:

| Name |  Hub URL | Description | Type | Category |  Test #Samples | 
|-----|-----|---------------------------|-----|-----|-----|
| [T2Retrieval](https://arxiv.org/abs/2304.03679) | [C-MTEB/T2Retrieval](https://huggingface.co/datasets/C-MTEB/T2Retrieval) |  T2Ranking: A large-scale Chinese Benchmark for Passage Ranking | Retrieval | s2p | 24,832 | 
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/MMarcoRetrieval](https://huggingface.co/datasets/C-MTEB/MMarcoRetrieval) | mMARCO is a multilingual version of the MS MARCO passage ranking dataset | Retrieval | s2p | 7,437 | 
| [DuRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/DuRetrieval](https://huggingface.co/datasets/C-MTEB/DuRetrieval) | A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine | Retrieval | s2p | 4,000 |
| [CovidRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/CovidRetrieval](https://huggingface.co/datasets/C-MTEB/CovidRetrieval) | COVID-19 news articles | Retrieval | s2p | 949  |
| [CmedqaRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/CmedqaRetrieval](https://huggingface.co/datasets/C-MTEB/CmedqaRetrieval) |  Online medical consultation text | Retrieval | s2p | 3,999 | 
| [EcomRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/EcomRetrieval](https://huggingface.co/datasets/C-MTEB/EcomRetrieval) | Passage retrieval dataset collected from Alibaba search engine systems in e-commerce domain | Retrieval | s2p | 1,000 |  
| [MedicalRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/MedicalRetrieval](https://huggingface.co/datasets/C-MTEB/MedicalRetrieval) | Passage retrieval dataset collected from Alibaba search engine systems in medical domain | Retrieval | s2p | 1,000  |
| [VideoRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/VideoRetrieval](https://huggingface.co/datasets/C-MTEB/VideoRetrieval) | Passage retrieval dataset collected from Alibaba search engine systems in video domain | Retrieval | s2p | 1,000  |
| [T2Reranking](https://arxiv.org/abs/2304.03679) | [C-MTEB/T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking) | T2Ranking: A large-scale Chinese Benchmark for Passage Ranking | Reranking | s2p | 24,382 | 
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/MMarco-reranking](https://huggingface.co/datasets/C-MTEB/Mmarco-reranking) | mMARCO is a multilingual version of the MS MARCO passage ranking dataset | Reranking | s2p | 7,437 | 
| [CMedQAv1](https://github.com/zhangsheng93/cMedQA) | [C-MTEB/CMedQAv1-reranking](https://huggingface.co/datasets/C-MTEB/CMedQAv1-reranking) | Chinese community medical question answering | Reranking | s2p |  2,000  |
| [CMedQAv2](https://github.com/zhangsheng93/cMedQA2) | [C-MTEB/CMedQAv2-reranking](https://huggingface.co/datasets/C-MTEB/C-MTEB/CMedQAv2-reranking) | Chinese community medical question answering | Reranking | s2p |  4,000  |
| [Ocnli](https://arxiv.org/abs/2010.05444) | [C-MTEB/OCNLI](https://huggingface.co/datasets/C-MTEB/OCNLI) | Original Chinese Natural Language Inference dataset | PairClassification | s2s |  3,000  |
| [Cmnli](https://huggingface.co/datasets/clue/viewer/cmnli) | [C-MTEB/CMNLI](https://huggingface.co/datasets/C-MTEB/CMNLI) | Chinese Multi-Genre NLI | PairClassification | s2s | 139,000  |
| [CLSClusteringS2S](https://arxiv.org/abs/2209.05034) | [C-MTEB/CLSClusteringS2S](https://huggingface.co/datasets/C-MTEB/C-MTEB/CLSClusteringS2S) | Clustering of titles from CLS dataset. Clustering of 13 sets, based on the main category. | Clustering | s2s |  10,000  |
| [CLSClusteringP2P](https://arxiv.org/abs/2209.05034) | [C-MTEB/CLSClusteringP2P](https://huggingface.co/datasets/C-MTEB/CLSClusteringP2P) | Clustering of titles + abstract from CLS dataset. Clustering of 13 sets, based on the main category. | Clustering | p2p | 10,000   |
| [ThuNewsClusteringS2S](http://thuctc.thunlp.org/) | [C-MTEB/ThuNewsClusteringS2S](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringS2S) | Clustering of titles from the THUCNews dataset | Clustering | s2s |  10,000  |
| [ThuNewsClusteringP2P](http://thuctc.thunlp.org/) | [C-MTEB/ThuNewsClusteringP2P](https://huggingface.co/datasets/C-MTEB/ThuNewsClusteringP2P) | Clustering of titles + abstract from the THUCNews dataset | Clustering | p2p |  10,000  |
| [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC) | [C-MTEB/ATEC](https://huggingface.co/datasets/C-MTEB/ATEC) | ATEC NLP sentence pair similarity competition | STS | s2s |  20,000  |
| [BQ](https://huggingface.co/datasets/shibing624/nli_zh) | [C-MTEB/BQ](https://huggingface.co/datasets/C-MTEB/BQ) | Bank Question Semantic Similarity | STS | s2s |  10,000  |
| [LCQMC](https://huggingface.co/datasets/shibing624/nli_zh) | [C-MTEB/LCQMC](https://huggingface.co/datasets/C-MTEB/LCQMC) | A large-scale Chinese question matching corpus. | STS | s2s |  12,500  |
| [PAWSX](https://arxiv.org/pdf/1908.11828.pdf) | [C-MTEB/PAWSX](https://huggingface.co/datasets/C-MTEB/PAWSX) | Translated PAWS evaluation pairs | STS | s2s |  2,000  |
| [STSB](https://github.com/pluto-junzeng/CNSD) | [C-MTEB/STSB](https://huggingface.co/datasets/C-MTEB/STSB) | Translate STS-B into Chinese | STS | s2s |  1,360  |
| [AFQMC](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/AFQMC](https://huggingface.co/datasets/C-MTEB/AFQMC) | Ant Financial Question Matching Corpus| STS | s2s |  3,861  |
| [QBQTC](https://github.com/CLUEbenchmark/QBQTC) | [C-MTEB/QBQTC](https://huggingface.co/datasets/C-MTEB/QBQTC) | QQ Browser Query Title Corpus | STS | s2s |  5,000  |
| [TNews](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/TNews-classification](https://huggingface.co/datasets/C-MTEB/TNews-classification) | Short Text Classificaiton for News | Classification | s2s |  10,000  |
| [IFlyTek](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/IFlyTek-classification](https://huggingface.co/datasets/C-MTEB/IFlyTek-classification) |  Long Text classification for the description of Apps | Classification | s2s |  2,600  |
| [Waimai](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb) | [C-MTEB/waimai-classification](https://huggingface.co/datasets/C-MTEB/waimai-classification) | Sentiment Analysis of user reviews on takeaway platforms | Classification | s2s |  1,000  |
| [OnlineShopping](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb) | [C-MTEB/OnlineShopping-classification](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification) | Sentiment Analysis of User Reviews on Online Shopping Websites | Classification | s2s |  1,000  |
| [MultilingualSentiment](https://github.com/tyqiangz/multilingual-sentiment-datasets) | [C-MTEB/MultilingualSentiment-classification](https://huggingface.co/datasets/C-MTEB/MultilingualSentiment-classification)  | A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative | Classification | s2s |  3,000  |
| [JDReview](https://huggingface.co/datasets/kuroneko5943/jd21) |  [C-MTEB/JDReview-classification](https://huggingface.co/datasets/C-MTEB/JDReview-classification) | review for iphone | Classification | s2s |  533  |

For retrieval tasks, we sample 100,000 candidates (including the ground truths) from the entire corpus to reduce the inference cost.  

## Acknowledgement

We thank the great tool from [Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb)  and the open-source datasets from Chinese NLP community.


## Citation

If you find this repository useful, please consider citation

```
@misc{c-pack,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
