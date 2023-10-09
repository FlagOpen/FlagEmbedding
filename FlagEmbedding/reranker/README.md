# Reranker

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

** We provide a [example](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/reranker/README.md) to fine-tune the reranker.**

## Train

This reranker is initialized from [xlm-roberta-base](https://huggingface.co/xlm-roberta-base), and we train it on a mixture of multilingual datasets:
- Chinese: 788,491 text pairs from [T2ranking](https://huggingface.co/datasets/THUIR/T2Ranking), [MMmarco](https://github.com/unicamp-dl/mMARCO), [dulreader](https://github.com/baidu/DuReader), and [nli-zh](https://huggingface.co/datasets/shibing624/nli_zh)
- English: 933,090 text pairs from [msmarco](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), [nq](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), [hotpotqa](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), and [NLI](https://github.com/princeton-nlp/SimCSE)
- Others: 97,458 text pairs from [Mr.TyDi](https://github.com/castorini/mr.tydi) (including arabic, bengali, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai)

In order to enhance the cross-language retrieval ability, we construct two cross-language retrieval datasets bases on [MMarco](https://github.com/unicamp-dl/mMARCO). 
Specifically, we sample 100,000 english queries to retrieve the chinese passages, and also sample 100,000 chinese queries to retrieve english passages.

Currently, this model mainly supports Chinese and English, and may see performance degradation for other low-resource languages.


## Evaluation

| Model | T2Reranking | T2RerankingZh2En\* | T2RerankingEn2Zh\* | MmarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| text2vec-base-multilingual | 64.66 | 62.94 | 62.51 | 14.37 | 48.46 | 48.6 | 50.26 |  
| multilingual-e5-small | 65.62 | 60.94 | 56.41 | 29.91 | 67.26 | 66.54 | 57.78 |  
| multilingual-e5-large | 64.55 | 61.61 | 54.28 | 28.6 | 67.42 | 67.92 | 57.4 |  
| multilingual-e5-base | 64.21 | 62.13 | 54.68 | 29.5 | 66.23 | 66.98 | 57.29 |  
| m3e-base | 66.03 | 62.74 | 56.07 | 17.51 | 77.05 | 76.76 | 59.36 |  
| m3e-large | 66.13 | 62.72 | 56.1 | 16.46 | 77.76 | 78.27 | 59.57 |  
| bge-base-zh-v1.5 | 66.49 | 63.25 | 57.02 | 29.74 | 80.47 | 84.88 | 63.64 |  
| bge-large-zh-v1.5 | 65.74 | 63.39 | 57.03 | 28.74 | 83.45 | 85.44 | 63.97 |  
| bge-reranker-base | 67.28 | 63.95 | 60.45 | 35.46 | 81.26 | 84.1 | 65.42 |  
| bge-reranker-large | 67.28 | 63.95 | 60.45 | 35.46 | 81.26 | 84.1 | 65.42 |  

\* : T2RerankingZh2En and T2RerankingEn2Zh are cross-language retrieval task
