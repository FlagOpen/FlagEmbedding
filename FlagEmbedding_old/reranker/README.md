# Reranker

## Usage 

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. 
You can get a relevance score by inputting query and passage to the reranker. 
The reranker is optimized based cross-entropy loss, so the relevance score is not bounded to a specific range.


### Using FlagEmbedding
```
pip install -U FlagEmbedding
```

Get relevance scores (higher scores indicate more relevance):
```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```


### Using Huggingface transformers

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


## Fine-tune

You can follow this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) to fine-tune the reranker.

This reranker is initialized from [xlm-roberta-base](https://huggingface.co/xlm-roberta-base), and we train it on a mixture of multilingual datasets:
- Chinese: 788,491 text pairs from [T2ranking](https://huggingface.co/datasets/THUIR/T2Ranking), [MMmarco](https://github.com/unicamp-dl/mMARCO), [dulreader](https://github.com/baidu/DuReader), [Cmedqa-v2](https://github.com/zhangsheng93/cMedQA2), and [nli-zh](https://huggingface.co/datasets/shibing624/nli_zh)
- English: 933,090 text pairs from [msmarco](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), [nq](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), [hotpotqa](https://huggingface.co/datasets/sentence-transformers/embedding-training-data), and [NLI](https://github.com/princeton-nlp/SimCSE)
- Others: 97,458 text pairs from [Mr.TyDi](https://github.com/castorini/mr.tydi) (including arabic, bengali, english, finnish, indonesian, japanese, korean, russian, swahili, telugu, thai)

In order to enhance the cross-language retrieval ability, we construct two cross-language retrieval datasets bases on [MMarco](https://github.com/unicamp-dl/mMARCO). 
Specifically, we sample 100,000 english queries to retrieve the chinese passages, and also sample 100,000 chinese queries to retrieve english passages.
The dataset has been released at [Shitao/bge-reranker-data](https://huggingface.co/datasets/Shitao/bge-reranker-data). 

Currently, this model mainly supports Chinese and English, and may see performance degradation for other low-resource languages.


## Evaluation

You can evaluate the reranker using our [c-mteb script](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB#evaluate-reranker)

| Model | T2Reranking | T2RerankingZh2En\* | T2RerankingEn2Zh\* | MmarcoReranking | CMedQAv1 | CMedQAv2 |  Avg  |  
|:-------------------------------|:-----------:|:------------------:|:------------------:|:---------------:|:--------:|:--------:|:-----:|  
| text2vec-base-multilingual |    64.66    |       62.94        |       62.51        |      14.37      |  48.46   |   48.6   | 50.26 |  
| multilingual-e5-small |    65.62    |       60.94        |       56.41        |      29.91      |  67.26   |  66.54   | 57.78 |  
| multilingual-e5-large |    64.55    |       61.61        |       54.28        |      28.6       |  67.42   |  67.92   | 57.4  |  
| multilingual-e5-base |    64.21    |       62.13        |       54.68        |      29.5       |  66.23   |  66.98   | 57.29 |  
| m3e-base |    66.03    |       62.74        |       56.07        |      17.51      |  77.05   |  76.76   | 59.36 |  
| m3e-large |    66.13    |       62.72        |        56.1        |      16.46      |  77.76   |  78.27   | 59.57 |  
| bge-base-zh-v1.5 |    66.49    |       63.25        |       57.02        |      29.74      |  80.47   |  84.88   | 63.64 |  
| bge-large-zh-v1.5 |    65.74    |       63.39        |       57.03        |      28.74      |  83.45   |  85.44   | 63.97 |  
| bge-reranker-base |    67.28    |       63.95        |       60.45        |      35.46      |  81.26   |   84.1   | 65.42 |  
| bge-reranker-large |    67.60    |       64.04        |       61.45        |      37.17      |  82.14   |  84.19   | 66.10 |  

\* : T2RerankingZh2En and T2RerankingEn2Zh are cross-language retrieval task



## Acknowledgement

Part of the code is developed based on [Reranker](https://github.com/luyug/Reranker).


## Citation

If you find this repository useful, please consider giving a star :star: and citation

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
