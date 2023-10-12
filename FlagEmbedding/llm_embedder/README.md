<div align="center">
<h1>LLM-Embedder [<a href="https://arxiv.org/abs/2310.07554">paper</a>]</h1>

<img src="imgs/llm-embedder.png" width="60%" class="center">
</div>

This is the codebase for LLM-Embedder, a unified embedding model to comprehensively support the retrieval augmentation needs of large language models, including knowledge retrieval, memory retrieval, examplar retrieval, and tool retrieval. It is fine-tuned over 6 tasks: 
- *Question Answering (qa)*
- *Conversational Search (convsearch)*
- *Long Conversation (chat)*
- *Long-Range Language Modeling (lrlm)*
- *In-Context Learning (icl)*
- *Tool Learning (tool)*

## Roadmap
- Details about how to fine-tune the LLM-Embedder are [here](docs/fine-tune.md).
- Details about how to evaluate different retrievers on various retrieval-augmented scenarios are [here](docs/evaluation.md).

## Usage
### Using `FlagEmbedding`
```pip install -U FlagEmbedding```
```python
from FlagEmbedding import LLMEmbedder

# Define queries and keys
queries = ["test query 1", "test query 2"]
keys = ["test key 1", "test key 2"]

# Load model (automatically use GPUs)
model = LLMEmbedder('BAAI/llm-embedder', use_fp16=False)

# Encode for a specific task (qa, icl, chat, lrlm, tool, convsearch)
task = "qa"
query_embeddings = model.encode_queries(queries, task=task)
key_embeddings = model.encode_keys(keys, task=task)

similarity = query_embeddings @ key_embeddings.T
print(similarity)
# [[0.8971, 0.8534]
# [0.8462, 0.9091]]
```


### Using `transformers`
```pip install -U transformers```
```python
import torch
from transformers import AutoTokenizer, AutoModel

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

# Define queries and keys
queries = ["test query 1", "test query 2"]
keys = ["test key 1", "test key 2"]

# Load model
tokenizer = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
model = AutoModel.from_pretrained('BAAI/llm-embedder')

# Add instructions for specific task (qa, icl, chat, lrlm, tool, convsearch)
instruction = INSTRUCTIONS["qa"]
queries = [instruction["query"] + query for query in queries]
keys = [instruction["key"] + key for key in keys]

# Tokenize sentences
query_inputs = tokenizer(queries, padding=True, return_tensors='pt')
key_inputs = tokenizer(keys, padding=True, return_tensors='pt')

# Encode
with torch.no_grad():
    query_outputs = model(**query_inputs)
    key_outputs = model(**key_inputs)
    # CLS pooling
    query_embeddings = query_outputs.last_hidden_state[:, 0]
    key_embeddings = key_outputs.last_hidden_state[:, 0]
    # Normalize
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    key_embeddings = torch.nn.functional.normalize(key_embeddings, p=2, dim=1)

similarity = query_embeddings @ key_embeddings.T
print(similarity)
# [[0.8971, 0.8534]
# [0.8462, 0.9091]]
```


### Using `sentence-transformers`
```pip install -U sentence-transformers```

```python
from sentence_transformers import SentenceTransformer

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

# Define queries and keys
queries = ["test query 1", "test query 2"]
keys = ["test key 1", "test key 2"]

# Load model
model = SentenceTransformer('BAAI/llm-embedder', device="cpu")

# Add instructions for specific task (qa, icl, chat, lrlm, tool, convsearch)
instruction = INSTRUCTIONS["qa"]
queries = [instruction["query"] + query for query in queries]
keys = [instruction["key"] + key for key in keys]

# Encode
query_embeddings = model.encode(queries)
key_embeddings = model.encode(keys)

similarity = query_embeddings @ key_embeddings.T
print(similarity)
# [[0.8971, 0.8534]
# [0.8462, 0.9091]]
```

## Contact
If you have any question or suggestion related to this project, feel free to open an issue or pull request. You also can email Peitian Zhang (namespace.pt@gmail.com).

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@misc{zhang2023retrieve,
      title={Retrieve Anything To Augment Large Language Models}, 
      author={Peitian Zhang and Shitao Xiao and Zheng Liu and Zhicheng Dou and Jian-Yun Nie},
      year={2023},
      eprint={2310.07554},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```