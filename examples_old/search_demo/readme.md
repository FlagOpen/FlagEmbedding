# Q&A Example

Vector Database can help LLMs to access external knowledge. 
You can load baai-general-embedding as the encoder to generate the vectors.
Here a example to build a bot which can answer your question using the knowledge in chinese wikipedia.

Here's a description of the Q&A dialogue scenario using flag embedding and a large language model:

1. **Data Preprocessing and Indexing:**
   - Download a Chinese wikipedia dataset.
   - Encode the Chinese wikipedia text using flag embedding.
   - Build an index using BM25.
2. **Query Enhancement with Large Language Model (LLM):**
   - Utilize a Large Language Model (LLM) to enhance and enrich the original user query based on the chat history.
   - The LLM can perform tasks such as text completion and paraphrasing to make the query more robust and comprehensive.
3. **Document Retrieval:**
   - Employ BM25 to retrieve the top-n documents from the locally stored Chinese wiki dataset based on the newly enhanced query.
4. **Embedding Retrieval:**
   - Perform an embedding retrieval on the top-n retrieved documents using brute force search to get top-k documents.
5. **Answer Retrieval with Language Model (LLM):**
   - Present the question, the top-k retrieved documents, and chat history to the Large Language Model (LLM).
   - The LLM can utilize its understanding of language and context to provide accurate and comprehensive answers to the user's question.

By following these steps, the Q&A system can leverage flag embedding, BM25 indexing, and a Large Language Model to improve the accuracy and intelligence of the system. The integration of these techniques can create a more sophisticated and reliable Q&A system for users, providing them with comprehensive information to effectively answer their questions.

### Installation

```shell
sudo apt install default-jdk
pip install -r requirements.txt
conda install -c anaconda openjdk
```

### Prepare Data

```shell
python pre_process.py --data_path ./data
```

This script will download the dataset (Chinese wikipedia), building BM25 index, inference embedding, and then save them to `data_path`.

## Q&A usage

### Run Directly

```shell
export OPENAI_API_KEY=...
python run.py --data_path ./data
```

This script will build a Q&A dialogue scenario.

### Quick Start

```python
# encoding=gbk
from tool import LocalDatasetLoader, BMVectorIndex, Agent
loader = LocalDatasetLoader(data_path="./data/dataset",
                            embedding_path="./data/emb/data.npy")
index = BMVectorIndex(model_path="BAAI/bge-large-zh",
                      bm_index_path="./data/index",
                      data_loader=loader)
agent = Agent(index)
question = "上次有人登月是什么时候"
agent.Answer(question, RANKING=1000, TOP_N=5, verbose=False)
```