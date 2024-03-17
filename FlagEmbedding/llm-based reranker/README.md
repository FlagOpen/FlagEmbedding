# Reranker

## Usage 

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. 
You can get a relevance score by inputting query and passage to the reranker. 
The reranker is optimized based cross-entropy loss, so the relevance score is not bounded to a specific range.


### Using FlagEmbedding
```
pip install -U FlagEmbedding
```

#### For normal reranker (bge-reranker-base / bge-reranker-large / bge-reranker-v2-m3 )

Get relevance scores (higher scores indicate more relevance):

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```

#### For LLM-based reranker

```python
from FlagEmbedding import FlagLLMReranker
reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_bf16=False) # Setting use_bf16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```

#### For LLM-based layerwise reranker

```python
from FlagEmbedding import LayerWiseFlagLLMReranker
reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_bf16=True) # Setting use_bf16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'], cutoff_layers=[28,40]) # Adjusting 'cutoff_layers' to pick which layers are used for computing the score."
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], cutoff_layers=[28,40])
print(scores)
```

### Using Huggingface transformers

#### For normal reranker (bge-reranker-base / bge-reranker-large / bge-reranker-v2-m3 )

Get relevance scores (higher scores indicate more relevance):

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```

#### For LLM-based reranker

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma')
yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = get_inputs(pairs, tokenizer)
    scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
    print(scores)
```

#### For LLM-based layerwise reranker

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'],
            sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-minicpm-layerwise', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to('cuda')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = get_inputs(pairs, tokenizer).to(model.device)
    all_scores = model(**inputs, return_dict=True, cutoff_layers=[28,40])
    all_scores = [scores[:, -1].view(-1, ).float() for scores in all_scores[0]]
    print(all_scores)
```


## Evaluation

Here are the evaluation results of BEIR. It rereank the top 100 results from bge-en-v1.5 large.

|             model              | inference length |    avg    | trec-covid | quora | dbpedia | scidocs | fever |  nq   | touche | arguana | msmarco | fiqa  | climae-fever | scifact | nfcorpus | hotpotqa | cqadupstack |
| :----------------------------: | :--------------: | :-------: | :--------: | :---: | :-----: | :-----: | :---: | :---: | :----: | :-----: | :-----: | :---: | :----------: | :-----: | :------: | :------: | :---------: |
|       bge-en-v1.5 large        |     512+512      |   54.31   |   74.89    | 89.06 |  44.16  |  22.62  | 87.17 | 55.04 | 25.08  |  63.54  |  42.48  | 44.97 |    36.49     |  74.64  |  38.12   |  74.11   |    42.23    |
|     mxbai-rerank-large-v1      |       1024       |   50.34   |   87.19    | 69.7  |  49.75  |  19.64  | 83.95 | 63.74 | 28.85  |  13.31  |  40.15  | 43.3  |    26.28     |  73.22  |  40.52   |  74.62   |    40.86    |
|      bce-reranker-base_v1      |       512        |   48.64   |   69.82    | 82.06 |  43.58  |  19.21  | 72.84 | 54.61 | 20.88  |  43.24  |  44.24  | 35.47 |    25.35     |  70.3   |  35.58   |  74.65   |    37.82    |
|     bge-reranker-v2-gemma      |       1024       |   60.71   |   85.51    | 90.37 |  49.92  |  21.65  | 90.8  | 72.6  | 35.68  |  78.68  |  48.07  | 49.32 |    39.07     |  77.22  |  39.73   |  86.15   |    45.85    |
|       bge-reranker-v2-m3       |       1024       |   55.36   |   83.39    | 89.13 |  48.15  |  18.25  | 90.15 | 69.37 | 33.22  |  37.7   |  47.79  | 44.51 |    37.99     |  73.08  |  34.85   |  84.51   |    38.24    |
|   bge-reranker-v2-minicpm-20   |       1024       |   56.4    |    79.8    | 89.59 |  48.87  |  20.68  | 89.62 | 66.41 | 25.55  |  66.16  |  47.01  | 42.15 |    27.77     |  75.19  |  39.98   |  83.88   |    43.33    |
|   bge-reranker-v2-minicpm-28   |       1024       |   59.86   |   84.97    | 90.21 |  51.12  |  21.58  | 90.15 | 72.54 | 28.03  |  78.81  |  48.1   | 48.05 |    33.57     |  77.35  |   41.1   |  85.74   |    46.56    |
| **bge-reranker-v2-minicpm-40** |       1024       | **59.92** |   84.49    | 90.12 |  51.03  |  21.53  | 90.37 | 72.54 | 28.65  |  78.26  |  48.28  | 48.26 |    34.11     |  77.36  |  41.27   |   85.8   |    46.8     |

Here are the evaluation results of BEIR. It rereank the top 100 results from e5 mistral 7b instruct.

|             model              | inference length |    Avg    | trec-covid | quora | dbpedia | scidocs | fever |  nq   | touche | arguana | msmarco | fiqa  | climate-fever | scifact | nfcorpus | hotpotqa | cqadupstack |
| :----------------------------: | ---------------- | :-------: | :--------: | :---: | :-----: | :-----: | :---: | :---: | :----: | :-----: | :-----: | :---: | :-----------: | :-----: | :------: | :------: | :---------: |
|     e5 mistral 7b instruct     | 512+512          |   56.85   |   87.07    | 89.59 |  48.84  |  16.3   | 87.82 | 63.56 | 26.24  |  61.8   |  43.06  | 56.58 |     38.37     |  76.26  |  38.58   |  75.72   |    42.97    |
|   bge-reranker-v2-minicpm-20   | 1024             |   57.11   |    85.8    | 89.6  |  50.61  |  19.91  | 90.27 | 66.97 | 25.22  |  66.3   |  46.97  | 43.22 |     27.2      |  76.37  |  39.55   |  84.76   |    43.85    |
| **bge-reranker-v2-minicpm-28** | 1024             | **60.43** |   88.99    | 90.22 |  52.05  |  20.93  | 90.89 | 73.22 | 28.87  |  79.19  |  48.19  | 49.05 |     32.67     |  77.98  |  40.65   |  86.56   |    47.06    |
|   bge-reranker-v2-minicpm-40   | 1024             |   60.41   |   87.82    | 90.16 |  52.09  |  20.91  | 91.11 | 73.19 | 29.04  |  78.57  |  48.33  | 49.22 |     33.28     |  77.84  |  40.66   |  86.64   |    47.34    |

Here are the evaluation results of CMTEB-retrieval. It rereank the top 100 results from bge-zh-v1.5 large.

|             model              | inference length |    avg    | T2Retrieval | MMarcoRetrieval | DuRetrieval | CovidRetrieval | CmedqaRetrieval | EcomRetrieval | MedicalRetrieval | VideoRetrieval |
| :----------------------------: | :--------------: | :-------: | :---------: | :-------------: | :---------: | :------------: | :-------------: | :-----------: | :--------------: | :------------: |
|       bge-zh-v1.5 large        |     512+512      |   70.46   |    83.99    |      79.23      |    86.31    |      73.4      |      42.57      |     65.32     |      59.55       |     73.31      |
|      bce-reranker-base_v1      |       512        |   60.71   |    73.85    |      78.3       |    77.08    |     80.69      |      36.78      |     44.31     |      53.32       |     41.36      |
|     bge-reranker-v2-gemma      |       1024       |   71.74   |    84.04    |      83.84      |    90.9     |     87.87      |      39.4       |     62.19     |      57.56       |     68.15      |
|       bge-reranker-v2-m3       |       1024       |   71.82   |    84.57    |      83.85      |    89.6     |     89.38      |      38.77      |     63.04     |      59.48       |     65.89      |
|   bge-reranker-v2-minicpm-20   |       1024       |   71.47   |    82.9     |      84.11      |    89.98    |     88.26      |      37.25      |     64.42     |      57.45       |     67.37      |
| **bge-reranker-v2-minicpm-28** |       1024       | **73.51** |    84.13    |      85.52      |    91.67    |     89.14      |      41.74      |     66.44     |      61.77       |     67.68      |
|   bge-reranker-v2-minicpm-40   |       1024       |   73.41   |    84.1     |      85.26      |    91.67    |     89.17      |      41.96      |     66.51     |      61.45       |     67.14      |

Here are the evaluation results of miracl (multi-language). It rereank the top 100 results from bge-m3.

| model                | inference length | avg       |  ar   |  bn   |  en   |  es   |  fa   |  fi   |  fr   |  hi   |  id   |  ja   |  ko   |  ru   |  sw   |  te   |  th   |  zh   |  de   |  yo   |
| -------------------- | :--------------: | --------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bge-m3               |     512+512      | 67.91     | 78.4  |  80   | 59.6  | 55.5  | 57.7  | 78.6  | 57.8  | 59.3  |  56   | 72.8  | 69.9  | 70.1  | 78.6  | 86.2  | 82.6  | 61.7  | 56.8  | 60.7  |
| m3 rerank            |       1024       | 72.84     | 81.7  | 84.63 | 63.45 | 63.71 | 62.49 | 82.41 | 63.26 | 68.25 | 62.71 | 79.96 | 73.79 | 76.93 | 82.27 | 89.36 | 85.3  | 64.2  | 62.64 | 64.03 |
| **gemma 2B**         |       1024       | **73.39** | 82.26 | 85.15 | 66.62 | 64.29 | 62.03 | 82.58 | 64.26 | 68.68 | 61.33 | 79.72 | 74.83 | 78.36 | 81.46 | 89.22 | 86.06 | 65.61 | 64.25 | 64.37 |
| minicpm - layer - 20 |       1024       | 62.26     | 71.8  | 62.15 | 62.51 | 56.45 | 45.28 | 74.73 | 52.77 | 50.64 | 57.3  | 70.57 | 69.12 | 60.58 | 67.4  | 76.98 | 67.87 | 61.61 | 49.74 | 63.2  |
| minicpm - layer - 28 |       1024       | 67.75     | 77.4  | 64.87 | 66.03 | 61.89 | 51.91 | 80.02 | 63.42 |  56   | 60.5  | 78.25 | 73.39 | 72.5  | 72.09 | 78.41 | 74.46 | 66.09 | 59.15 | 63.07 |
| minicpm - layer - 40 |       1024       | 67.77     | 77.49 | 64.8  | 66.02 | 62.05 | 51.78 | 80.12 | 62.98 | 55.7  | 60.51 | 78.07 | 73.01 | 72.64 | 72.81 | 78.51 | 74.24 | 65.82 | 59.43 | 63.91 |

Here are the evaluation results of llama-index.

|      embedding model       | bge-en-v1.5 large | bge-en-v1.5 large |  bge-m3   |    bge-m3    | openai-small | openai-small | openai-large | openai-large | mxbai-embed-large-v1 | mxbai-embed-large-v1 |
| :------------------------: | :---------------: | :---------------: | :-------: | :----------: | :----------: | :----------: | :----------: | :----------: | :------------------: | :------------------: |
|        **reranker**        |      **mrr**      |   **hit rate**    |  **mrr**  | **hit rate** |   **mrr**    | **hit rate** |   **mrr**    | **hit rate** |       **mrr**        |     **hit rate**     |
|      without reranker      |       65.07       |       85.1        |   69.67   |    88.94     |    65.69     |    89.42     |    67.37     |    90.38     |        66.66         |        88.46         |
|     bge-reranekr-base      |       75.77       |       90.87       |   77.48   |    94.23     |    75.75     |    93.27     |     76.3     |    94.23     |        76.63         |        91.83         |
|     bge-reranker-large     |       75.86       |       90.87       |   78.66   |    94.23     |    77.09     |    94.23     |    77.08     |  **95.67**   |        77.24         |        92.31         |
|   mxbai-rerank-large-v1    |       72.77       |       88.46       |   75.99   |    93.27     |    74.62     |    91.35     |    74.32     |    92.31     |        73.89         |         89.9         |
|  jina-reranker-v1-base-en  |       75.81       |       89.9        |   79.44   |    93.75     |    77.64     |    91.83     |    77.85     |    92.79     |        76.96         |        91.83         |
|       cohere rerank        |       75.17       |       90.38       |   76.23   |    91.35     |    76.98     |    92.79     |    76.68     |    93.27     |        76.43         |        92.31         |
|   ms-marco-MiniLM-L-6-v2   |       67.92       |       86.54       |   69.83   |    90.38     |     69.2     |    88.46     |    67.99     |    90.38     |        68.57         |        87.98         |
|     bge-reranker-v2-m3     |       78.26       |       90.87       |   80.76   |    94.71     |    79.38     |    93.27     |     79.7     |    94.71     |         79.1         |        92.31         |
|   bge-reranker-v2-gemma    |       75.19       |       89.9        |   78.14   |    93.75     |    76.74     |    92.31     |    76.28     |    92.31     |        77.25         |        91.83         |
| bge-reranker-v2-minicpm-20 |       81.31       |     **91.83**     |   83.77   |  **95.67**   |    81.92     |  **94.71**   |    83.43     |    95.19     |        82.11         |        92.79         |
| bge-reranker-v2-minicpm-28 |     **81.93**     |     **91.83**     | **84.74** |  **95.67**   |  **84.01**   |  **94.71**   |  **83.93**   |    95.19     |      **82.99**       |      **93.27**       |
| bge-reranker-v2-minicpm-40 |       80.89       |     **91.83**     |   83.29   |  **95.67**   |    82.89     |  **94.71**   |    82.33     |    95.19     |        81.45         |        92.79         |

## Acknowledgement

Part of the code is developed based on [Reranker](https://github.com/luyug/Reranker).
