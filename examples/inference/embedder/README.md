# Embedder

- [Model List](#model-list)
- [Usage](#usage)
- [Citation](#citation)

An embedder can encode text into embeddings.

When provided with a query and a passage, the embedder encodes both separately, and then uses the similarity between their embeddings as the similarity score.

For more detailed using, you can look [embedder-encoder only](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder/encoder_only) or [embedder-decoder only](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder/decoder_only)


## Model List

`bge` is short for `BAAI general embedding`.

| Model                                                        |      Language       |                         Description                          |               query instruction for retrieval                |
| :----------------------------------------------------------- | :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl)    |       English       | A LLM-based embedding model with in-context learning capabilities, which can fully leverage the model's potential based on a few shot examples | Provide instructions and few-shot examples freely based on the given task. |
| [BAAI/bge-multilingual-gemma2](https://huggingface.co/BAAI/bge-multilingual-gemma2) |    Multilingual     | A LLM-based multilingual embedding model, trained on a diverse range of languages and tasks. |        Provide instructions based on the given task.         |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)            |    Multilingual     | Multi-Functionality(dense retrieval, sparse retrieval, multi-vector(colbert)), Multi-Linguality, and Multi-Granularity(8192 tokens) |                                                              |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |       English       |   version 1.5 with more reasonable similarity distribution   | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |       English       |   version 1.5 with more reasonable similarity distribution   | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |       English       |   version 1.5 with more reasonable similarity distribution   | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |       Chinese       |   version 1.5 with more reasonable similarity distribution   |           `为这个句子生成表示以用于检索相关文章：`           |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |       Chinese       |   version 1.5 with more reasonable similarity distribution   |           `为这个句子生成表示以用于检索相关文章：`           |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |       Chinese       |   version 1.5 with more reasonable similarity distribution   |           `为这个句子生成表示以用于检索相关文章：`           |
| [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |       English       |          Embedding Model which map text into vector          | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en)  |       English       | a base-scale model but with similar ability to `bge-large-en` | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |       English       |     a small-scale model but with competitive performance     | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |       Chinese       |          Embedding Model which map text into vector          |           `为这个句子生成表示以用于检索相关文章：`           |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)  |       Chinese       | a base-scale model but with similar ability to `bge-large-zh` |           `为这个句子生成表示以用于检索相关文章：`           |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |       Chinese       |     a small-scale model but with competitive performance     |           `为这个句子生成表示以用于检索相关文章：`           |

## Usage

### Using FlagEmbedding

#### 1. Auto Model

You can use `FlagAutoModel` to load the model. For the **custom model** (not included in [`AUTO_EMBEDDER_MAPPING`](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/model_mapping.py#L39)), you must specify the `model_class` parameter. You can also submit a pull request to add your **released model** to the [`AUTO_EMBEDDER_MAPPING`](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/model_mapping.py#L39) dictionary. If need, you can create a new `<model>.py` file in [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/encoder_only) or [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/inference/embedder/decoder_only).

```python
from FlagEmbedding import FlagAutoModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
```

For your **custom model** (assume the model is finetuned from `BAAI/bge-large-zh-v1.5`, then the model class is `encoder-only-base`), you can use the following code:

```python
from FlagEmbedding import FlagAutoModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagAutoModel.from_finetuned('your_model_name_or_path',
                                     model_class='encoder-only-base',   # specify the model class
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     pooling_method='cls',  # specify the pooling method
                                     use_fp16=True,
                                     devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
```

The `model_class` parameter currently includes the following options:
- `encoder-only-base`: for encoder-only normal model, such as `BAAI/bge-large-en-v1.5`
- `encoder-only-m3`: for encoder-only M3 model, such as `BAAI/bge-m3`
- `decoder-only-base`: for decoder-only normal model, such as `BAAI/bge-multilingual-gemma2`
- `decoder-only-icl`: for decoder-only ICL model, such as `BAAI/bge-en-icl`

#### 2. Normal Model

For `FlagModel`, it supports `BAAI/bge-large-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-large-zh-v1.5`, `BAAI/bge-base-zh-v1.5`, `BAAI/bge-small-zh-v1.5`, `BAAI/bge-large-en`, `BAAI/bge-base-en`, `BAAI/bge-small-en`, `BAAI/bge-large-zh`, `BAAI/bge-base-zh`, `BAAI/bge-small-zh'`:

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5',
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True,
                  devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
```

#### 3. M3 Model

For `BGEM3FlagModel`, it supports `BAAI/bge-m3`:

```python
from FlagEmbedding import BGEM3FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True,
                       pooling_method='cls',
                       devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(
    sentences_1,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
embeddings_2 = model.encode(
    sentences_2,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
dense_similarity = embeddings_1["dense_vecs"] @ embeddings_2["dense_vecs"].T
print('dense similarity:', dense_similarity)
sparse_similarity = model.compute_lexical_matching_score(
    embeddings_1["lexical_weights"],
    embeddings_2["lexical_weights"],
)
print('sparse similarity:', sparse_similarity)

queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(
    queries,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
p_embeddings = model.encode_corpus(
    passages,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
dense_scores = embeddings_1["dense_vecs"] @ embeddings_2["dense_vecs"].T
print('dense scores:', dense_scores)
sparse_scores = model.compute_lexical_matching_score(
    embeddings_1["lexical_weights"],
    embeddings_2["lexical_weights"],
)
print('sparse similarity:', sparse_scores)
```

#### 4. LLM-based Model

For `FlagLLMModel`, it supports `BAAI/bge-multilingual-gemma2`, `Alibaba-NLP/gte-Qwen2-7B-instruct`, `intfloat/e5-mistral-7b-instruct`, .etc:

```python
from FlagEmbedding import FlagLLMModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagLLMModel('BAAI/bge-multilingual-gemma2',
                     query_instruction_for_retrieval="Given a question, retrieve passages that answer the question.",
                     query_instruction_format="<instruct>{}\n<query>{}",
                     use_fp16=True,
                     devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
```

#### 5. LLM-based ICL Model

For `FlagICLModel`, it supports `BAAI/bge-en-icl`:

```python
from FlagEmbedding import FlagICLModel

examples = [
    {
        'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
        'query': 'what is a virtual interface',
        'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."
    },
    {
        'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
        'query': 'causes of back pain in female for a week',
        'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."
    }
]
model = FlagICLModel(
    'BAAI/bge-en-icl',
    query_instruction_for_retrieval="Given a question, retrieve passages that answer the question.",
    query_instruction_format="<instruct>{}\n<query>{}",
    examples_for_task=examples,
    examples_instruction_format="<instruct>{}\n<query>{}\n<response>{}",
    use_fp16=True,
    devices=['cuda:1']
) # Setting use_fp16 to True speeds up computation with a slight performance degradation
queries = [
    "how much protein should a female eat",
    "summit define"
]
passages = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)
```

### Using HuggingFace Transformers

#### 1. Normal Model

It supports `BAAI/bge-large-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-large-zh-v1.5`, `BAAI/bge-base-zh-v1.5`, `BAAI/bge-small-zh-v1.5`, `BAAI/bge-large-en`, `BAAI/bge-base-en`, `BAAI/bge-small-en`, `BAAI/bge-large-zh`, `BAAI/bge-base-zh`, `BAAI/bge-small-zh'`, the **dense method** of `BAAI/bge-m3`:

```python
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
with torch.no_grad():
    encoded_input_1 = tokenizer(sentences_1, padding=True, truncation=True, return_tensors='pt')
    encoded_input_2 = tokenizer(sentences_2, padding=True, truncation=True, return_tensors='pt')
    model_output_1 = model(**encoded_input_1)
    model_output_2 = model(**encoded_input_2)
    embeddings_1 = model_output_1[0][:, 0]
    embeddings_2 = model_output_2[0][:, 0]
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
```

#### 2. M3 Model

It only supports the **dense method** of `BAAI/bge-m3`, you can refer to the above code.

#### 3. LLM-based Model

It supports `BAAI/bge-multilingual-gemma2`:

```python
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'


task = 'Given a web search query, retrieve relevant passages that answer the query.'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
# No need to add instructions for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-multilingual-gemma2')
model = AutoModel.from_pretrained('BAAI/bge-multilingual-gemma2')
model.eval()

max_length = 4096
# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8)

with torch.no_grad():
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# [[55.92064666748047, 1.6549524068832397], [-0.2698777914047241, 49.95653533935547]]
```

#### 4. LLM-based ICL Model

It supports `BAAI/bge-en-icl`:

```python
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'

def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries

task = 'Given a web search query, retrieve relevant passages that answer the query.'
examples = [
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'what is a virtual interface',
   'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
  {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
   'query': 'causes of back pain in female for a week',
   'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
]
examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]
query_max_len, doc_max_len = 512, 512

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-en-icl')
model = AutoModel.from_pretrained('BAAI/bge-en-icl')
model.eval()

new_query_max_len, new_queries = get_new_queries(queries, query_max_len, examples_prefix, tokenizer)

query_batch_dict = tokenizer(new_queries, max_length=new_query_max_len, padding=True, truncation=True, return_tensors='pt')
doc_batch_dict = tokenizer(documents, max_length=doc_max_len, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    query_outputs = model(**query_batch_dict)
    query_embeddings = last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    doc_outputs = model(**doc_batch_dict)
    doc_embeddings = last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])
    
# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
scores = (query_embeddings @ doc_embeddings.T) * 100
print(scores.tolist())
```

### Using Sentence-Transformers

You can also use the `bge` models with [sentence-transformers](https://www.sbert.net/). It currently supports `BAAI/bge-large-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-large-zh-v1.5`, `BAAI/bge-base-zh-v1.5`, `BAAI/bge-small-zh-v1.5`, `BAAI/bge-large-en`, `BAAI/bge-base-en`, `BAAI/bge-small-en`, `BAAI/bge-large-zh`, `BAAI/bge-base-zh`, `BAAI/bge-small-zh'`, the **dense method** of `BAAI/bge-m3`, `BAAI/bge-multilingual-gemma2`:

```
pip install -U sentence-transformers
```

```shell
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

For s2p(short query to long passage) retrieval task, each short query should start with an instruction (instructions see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). But the instruction is not needed for passages.

```shell
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

### Using Langchain

You can use `bge` in langchain like this:

```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："
```

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
@misc{bge-m3,
      title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation}, 
      author={Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2402.03216},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{li2024makingtextembeddersfewshot,
      title={Making Text Embedders Few-Shot Learners}, 
      author={Chaofan Li and MingHao Qin and Shitao Xiao and Jianlyu Chen and Kun Luo and Yingxia Shao and Defu Lian and Zheng Liu},
      year={2024},
      eprint={2409.15700},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2409.15700}, 
}
```

