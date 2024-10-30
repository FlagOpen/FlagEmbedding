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

With the transformers package, you can use the model like this: First, you pass your input through the transformer model, then you select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

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


### Using Sentence-Transformers

You can also use the `bge` models with [sentence-transformers](https://www.sbert.net/):

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

