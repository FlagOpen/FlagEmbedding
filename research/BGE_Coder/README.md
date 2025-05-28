<h1 align="center">CodeR: Towards A Generalist Code Embedding Model</h1>
<p align="center">
    <a href="https://huggingface.co/datasets/nebula2025/CodeR-Pile">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-CodeR Pile-yellow">
    </a>
    <a href="https://huggingface.co/nebula2025/CodeR-full">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-CodeR Full-green">
    </a>
    <a href="https://huggingface.co/nebula2025/CodeR-synthetic">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-CodeR Synthetic-blue">
    </a>
</p>


This repo contains the data, training, and evaluation pipeline for CodeR / [BGE-Code-v1](https://huggingface.co/BAAI/bge-code-v1)

**[BGE-Code-v1](https://huggingface.co/BAAI/bge-code-v1)** is an LLM-based code embedding model that supports code retrieval, text retrieval, and multilingual retrieval. It primarily demonstrates the following capabilities:

- Superior Code Retrieval Performance: The model demonstrates exceptional code retrieval capabilities, supporting natural language queries in both English and Chinese, as well as 20 programming languages.
- Robust Text Retrieval Capabilities: The model maintains strong text retrieval capabilities comparable to text embedding models of similar scale.
- Extensive Multilingual Support: BGE-Code-v1 offers comprehensive multilingual retrieval capabilities, excelling in languages such as English, Chinese, Japanese, French, and more.

## :bell: News:

- 🥳 5/15/2025: We have released the CodeR! :fire:

## Usage

### Using FlagEmbedding

```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
from FlagEmbedding import FlagLLMModel
queries = [
    "Delete the record with ID 4 from the 'Staff' table.", 
    'Delete all records in the "Livestock" table where age is greater than 5'
]
documents = [
    "DELETE FROM Staff WHERE StaffID = 4;",
    "DELETE FROM Livestock WHERE age > 5;"
]
model = FlagLLMModel('BAAI/bge-code-v1', 
                     query_instruction_format="<instruct>{}\n<query>{}",
                     query_instruction_for_retrieval="Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
                     trust_remote_code=True,
                     use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode_queries(queries)
embeddings_2 = model.encode_corpus(documents)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

By default, FlagLLMModel will use all available GPUs when encoding. Please set `os.environ["CUDA_VISIBLE_DEVICES"]` to select specific GPUs. You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make all GPUs unavailable.

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
import torch

# Load the model, optionally in float16 precision for faster inference
model = SentenceTransformer(
    "BAAI/bge-code-v1",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.float16},
)

# Prepare a prompt given an instruction
instruction = 'Given a question in text, retrieve SQL queries that are appropriate responses to the question.'
prompt = f'<instruct>{instruction}\n<query>'
# Prepare queries and documents
queries = [
    "Delete the record with ID 4 from the 'Staff' table.", 
    'Delete all records in the "Livestock" table where age is greater than 5'
]
documents = [
    "DELETE FROM Staff WHERE StaffID = 4;",
    "DELETE FROM Livestock WHERE age > 5;"
]

# Compute the query and document embeddings
query_embeddings = model.encode(queries, prompt=prompt)
document_embeddings = model.encode(documents)

# Compute the cosine similarity between the query and document embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
```

### Using HuggingFace Transformers

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


instruction = 'Given a question in text, retrieve SQL queries that are appropriate responses to the question.'
queries = [
    "Delete the record with ID 4 from the 'Staff' table.", 
    'Delete all records in the "Livestock" table where age is greater than 5'
]
documents = [
    "DELETE FROM Staff WHERE StaffID = 4;",
    "DELETE FROM Livestock WHERE age > 5;"
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-code-v1', trust_remote_code=True)
model = AutoModel.from_pretrained('BAAI/bge-code-v1', trust_remote_code=True)
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
```

## Evaluation

**BGE-Code-v1** achieves state-of-the-art performance on both the CoIR and CodeRAG benchmarks.

- CoIR

|                       | CodeXEmbed-2B | CodeXEmbed-7B | Voyage-Code-002 | Voyage-Code-003 | BGE-Code-v1 |
| --------------------- | ------------- | ------------- | --------------- | --------------- | ----------- |
| **Apps**              | 76.86         | 85.38         | 26.52           | 93.62           | 98.08       |
| **CosQA**             | 40.47         | 42.47         | 29.79           | 34.45           | 46.72       |
| **Text2SQL**          | 78.42         | 78.94         | 69.26           | 62.87           | 64.35       |
| **CSN**               | 87.87         | 89.67         | 81.79           | 89.35           | 89.53       |
| **CSN-CCR**           | 97.66         | 97.95         | 73.45           | 90.05           | 98.30       |
| **CodeTrans-Contest** | 90.30         | 94.45         | 72.77           | 94.96           | 94.38       |
| **CodeTrans-DL**      | 38.57         | 40.46         | 27.48           | 38.57           | 46.13       |
| **StackOverFlow-QA**  | 94.47         | 96.33         | 67.68           | 97.17           | 95.35       |
| **CodeFeedBack-ST**   | 86.36         | 87.53         | 65.35           | 90.67           | 90.56       |
| **CodeFeedBack-MT**   | 65.51         | 68.83         | 28.74           | 93.58           | 94.38       |
| **AVG**               | **75.65**     | **78.20**     | **56.26**       | **78.53**       | **81.77**   |

- CodedRAG

|                 | HummanEval | MBPP | DS-1000 | ODEX | RepoEval | SWE-bench-Lite | AVG      |
| --------------- | ---------- | ---- | ------- | ---- | -------- | -------------- | -------- |
| SFR             | 100.0      | 99.0 | 19.3    | 37.1 | 83.8     | 62.7           | **67.0** |
| Jina-v2-code    | 100.0      | 97.7 | 26.2    | 19.9 | 90.5     | 58.3           | **65.4** |
| CodeXEmbed-2B   | 100.0      | 97.4 | 25.4    | 23.9 | 88.7     | 52.4           | **64.6** |
| Voyage-Code-002 | 100.0      | 99.0 | 33.1    | 26.6 | 94.3     | 29.1           | **63.7** |
| BGE-Code-v1     | 100.0      | 99.2 | 40.9    | 36.1 | 93.1     | 67.4           | **72.8** |

### Instructions for Evaluation

```python
{
    "Apps": "Given a code contest problem description, retrieve relevant code that can help solve the problem.",
    "CosQA": "Given a web search query, retrieve relevant code that can help answer the query.",
    "Text2SQL": "Given a question in text, retrieve SQL queries that are appropriate responses to the question.",
    "CSN": "Given a piece of code, retrieve the document string that summarizes the code.",
    "CSN-CCR": "Given a piece of code segment, retrieve the code segment that is the latter part of the code.",
    "CodeTrans-DL": "Given a piece of code, retrieve code that is semantically equivalent to the input code.",
    "CodeTrans-Contest": "Given a piece of Python code, retrieve C++ code that is semantically equivalent to the input code.",
    "StackOverFlow-QA": "Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "CodeFeedBack-ST": "Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "CodeFeedBack-MT": "Given a multi-turn conversation history that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "HummanEval": "Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "MBPP": "Given a textual explanation of code functionality, retrieve the corresponding code implementation.",
    "DS-1000": "Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "ODEX": "Given a question, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.",
    "RepoEval": "Given a piece of code segment, retrieve the code segment that is the latter part of the code.",
    "SWE-bench-Lite": "Given a code snippet containing a bug and a natural language description of the bug or error, retrieve code snippets that demonstrate solutions or fixes for similar bugs or errors (the desired documents)."
}
```

### Evaluation script

#### CoIR

For CoIR, we use the [CoIR](https://github.com/CoIR-team/coir) evaluation script:

```shell
cd ./evaluation/coir_eval
### clone coir
mkdir test
cd ./test
git clone https://github.com/CoIR-team/coir.git
mv ./coir/coir ../
cd ..
rm -rf ./test
### evaluate
bash eval.sh
```

### CodeRAG

For CodeRAG, we use the [CodeRAG](https://github.com/code-rag-bench/code-rag-bench) evaluation script:

```shell
cd ./evaluation/coderag_eval
### clone coderag
git clone https://github.com/code-rag-bench/code-rag-bench.git
## You need prepare environment according to README.md
rm -rf ./code-rag-bench/retrieval/create
cp -r ./test/* ./code-rag-bench/retrieval/
### prepare data
bash prepare_data.sh
### evaluate
bash eval.sh
```