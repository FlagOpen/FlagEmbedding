# Chinese Massive Text Embedding Benchmark 


## Installation
C-MTEB is devloped based on [MTEB](https://github.com/embeddings-benchmark/mteb). 
```
pip install mteb[beir]
```

## Usage
* python script  
You can use C-MTEB easily in the same way as [MTEB](https://github.com/embeddings-benchmark/mteb).

```python
from mteb import MTEB
from C_MTEB import *
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "bert-base-uncased"

model = SentenceTransformer(model_name)
evaluation = MTEB(task_langs=['zh'])
results = evaluation.run(model, output_folder=f"results/{model_name}")
```

* Reproduce the results of flag_embedding  
Using the provided python script (see [eval_C-MTEB.py]() )
```bash
python eval_C-MTEB.py --model_name_or_path BAAI/bge-large-zh
```

* Using a custom model  
To evaluate a new model, you can load it by sentence_transformers if it is supported by sentence_transformers.
Otherwise,  models should be implemented like this (implementing an `encode` function taking as inputs a list of sentences, and returning a list of embeddings (embeddings can be `np.array`, `torch.tensor`, etc.). 
): 

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

### overall
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**baai-general-embedding-large-zh-instruction**](https://huggingface.co/BAAI/bge-large-zh) | 1024 | **64.20** | **71.53** | **53.23** | **78.94** | 72.26 | **65.11** | 48.39 |  
| [baai-general-embedding-large-zh](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 50.98 | 76.77 | **72.49** | 64.91 | **50.01** |   
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 |56.91 | 48.15 | 63.99 | 70.28 | 59.34 | 47.68 |  
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 |54.75 | 48.64 | 64.3 | 71.22 | 59.66 | 48.88 |  
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 40.61 | 69.56 | 67.38 | 54.28 | 45.68 |  
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 39.41 | 66.62 | 65.29 | 49.25 | 44.39 | 
| [text2vec](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 41.71 | 67.41 | 65.18 | 49.45 | 37.66 |  
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 41.98 | 70.86 | 63.42 | 49.16 | 30.02 |  



### 1. Retrieval
| Model | T2Retrieval | MMarcoRetrieval | DuRetrieval | CovidRetrieval | CmedqaRetrieval | EcomRetrieval | MedicalRetrieval | VideoRetrieval | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 58.67 | 55.31 | 59.36 | 55.48 | 18.04 | 40.48 | 29.8 | 38.04 | 44.4 |  
| text2vec-large-chinese | 50.52 | 45.96 | 51.87 | 60.48 | 15.53 | 37.58 | 30.93 | 42.65 | 41.94 |  
| text2vec-base-chinese | 51.67 | 44.06 | 52.23 | 44.81 | 15.91 | 34.59 | 27.56 | 39.52 | 38.79 |  
| m3e-base | 73.14 | 65.45 | 75.76 | 66.42 | 30.33 | 50.27 | 42.8 | 51.11 | 56.91 |  
| m3e-large | 72.36 | 61.06 | 74.69 | 61.33 | 30.73 | 45.18 | 48.66 | 44.02 | 54.75 |  
| OpenAI(text-embedding-ada-002) | 69.14 | 69.86 | 71.17 | 57.21 | 22.36 | 44.49 | 37.92 | 43.85 | 52.0 |  
| universal-embedding | 84.39 | 81.38 | 84.68 | 75.07 | 41.03 | 65.6 | 58.28 | 73.94 | 70.55 |  
| universal-embedding-instruction | 84.82 | 81.28 | 86.94 | 74.06 | 42.4 | 66.12 | 59.39 | 77.19 | 71.53 |  


### 2.  STS  
| Model | ATEC | BQ | LCQMC | PAWSX | STSB | AFQMC | QBQTC | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 30.84 | 43.33 | 66.74 | 12.31 | 73.22 | 22.24 | 27.2 | 39.41 |  
| text2vec-large-chinese | 32.45 | 44.22 | 69.16 | 14.55 | 79.45 | 24.51 | 29.51 | 41.98 |  
| text2vec-base-chinese | 31.93 | 42.67 | 70.16 | 17.21 | 79.3 | 26.06 | 24.62 | 41.71 | 
| m3e-base | 41.27 | 63.81 | 74.88 | 12.19 | 76.97 | 35.87 | 32.07 | 48.15 |   
| m3e-large | 41.8 | 65.2 | 74.2 | 15.95 | 74.16 | 36.53 | 32.65 | 48.64 |  
| OpenAI(text-embedding-ada-002) | 29.25 | 45.33 | 68.41 | 16.55 | 70.61 | 23.88 | 30.27 | 40.61 |  
| universal-embedding | 48.29 | 60.53 | 74.71 | 16.64 | 78.41 | 43.06 | 35.2 | 50.98 |  
| universal-embedding-instruction | 49.75 | 62.93 | 75.45 | 22.45 | 78.51 | 44.57 | 38.92 | 53.23 |  


### 3. PairClassification  
| Model | Ocnli | Cmnli | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 60.7 | 72.55 | 66.62 |  
| text2vec-large-chinese | 64.04 | 77.67 | 70.86 |  
| text2vec-base-chinese | 60.95 | 73.87 | 67.41 |  
| m3e-base | 58.0 | 69.98 | 63.99 |  
| m3e-large | 59.33 | 69.27 | 64.3 |  
| OpenAI(text-embedding-ada-002) | 63.08 | 76.03 | 69.56 |  
| universal-embedding | 71.37 | 82.17 | 76.77 |  
| universal-embedding-instruction | 75.75 | 82.12 | 78.94 |  


### 4. Classification  
| Model | TNews | IFlyTek | MultilingualSentiment | JDReview | OnlineShopping | Waimai | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 45.22 | 41.75 | 61.21 | 79.68 | 84.3 | 79.57 | 65.29 |  
| text2vec-large-chinese | 38.92 | 41.54 | 58.97 | 81.56 | 83.51 | 76.01 | 63.42 |  
| text2vec-base-chinese | 43.02 | 42.05 | 60.98 | 82.14 | 85.69 | 77.22 | 65.18 |  
| m3e-base | 48.28 | 44.42 | 71.9 | 85.33 | 87.77 | 83.99 | 70.28 |  
| m3e-large | 48.26 | 43.96 | 72.47 | 86.92 | 89.59 | 86.1 | 71.22 |  
| OpenAI(text-embedding-ada-002) | 45.77 | 44.62 | 67.99 | 74.6 | 88.94 | 82.37 | 67.38 |  
| universal-embedding | 52.05 | 45.32 | 73.7 | 85.38 | 91.66 | 86.83 | 72.49 |  
| universal-embedding-instruction | 50.84 | 45.09 | 74.41 | 85.08 | 91.6 | 86.54 | 72.26 |  


### 5. Reranking  
| Model | T2Reranking | MmarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 65.76 | 14.55 | 57.82 | 58.88 | 49.25 |  
| text2vec-large-chinese | 64.82 | 12.48 | 58.92 | 60.41 | 49.16 |  
| text2vec-base-chinese | 65.95 | 12.76 | 59.26 | 59.82 | 49.45 | 
| m3e-base | 66.03 | 17.51 | 77.05 | 76.76 | 59.34 |   
| m3e-large | 66.13 | 16.46 | 77.76 | 78.27 | 59.66 |  
| OpenAI(text-embedding-ada-002) | 66.65 | 23.39 | 63.08 | 64.02 | 54.28 |  
| baai-general-embedding-large-zh | 66.16 | 27.1 | 81.72 | 84.64 | 64.91 |  
| baai-general-embedding-large-zh-instruction | 66.19 | 26.23 | 83.01 | 85.01 | 65.11 |  

### 6. Clustering  
| Model | CLSClusteringS2S | CLSClusteringP2P | ThuNewsClusteringS2S | ThuNewsClusteringP2P | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|  
| luotuo-bert-medium | 33.46 | 37.01 | 48.26 | 58.83 | 44.39 |  
| text2vec-large-chinese | 28.77 | 30.13 | 26.14 | 35.05 | 30.02 |  
| text2vec-base-chinese | 32.42 | 35.27 | 40.01 | 42.92 | 37.66 |
| m3e-base | 37.34 | 39.81 | 53.78 | 59.77 | 47.68 |    
| m3e-large | 38.02 | 38.6 | 58.51 | 60.39 | 48.88 |  
| OpenAI(text-embedding-ada-002) | 35.91 | 38.26 | 49.86 | 58.71 | 45.68 |  
| universal-embedding | 40.04 | 41.23 | 56.75 | 62.03 | 50.01 |  
| universal-embedding-instruction | 38.05 | 40.92 | 58.79 | 55.79 | 48.39 |  



## Tasks

An overview of tasks and datasets available in MTEB-chinese is provided in following table:

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
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/Mmarco-reranking](https://huggingface.co/datasets/C-MTEB/Mmarco-reranking) | mMARCO is a multilingual version of the MS MARCO passage ranking dataset | Reranking | s2p | 7,437 | 
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

In retrieval task, we sample 100,000 candidates (including the ground truths) from entire corpus to reduce the inference cost.  

## Acknowledgement
This work is inspired by [Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb), 
which lacks of the evaluation for chinese text.


