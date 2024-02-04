# MultiLongDocRetrieval

MultiLongDocRetrieval (denoted as MLDR) is a multilingual long-document retrieval dataset.

## Dense Retrieval

If you want to evaluate **embedding models**, you can use [this script](./eval_MLDR.py). The following is an example:

```bash
python3 eval_MLDR.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--results_save_path ./results \
--cache_dir /home/datasets/.cache \
--max_query_length 512 \
--max_passage_length 8192 \
--batch_size 256 \
--corpus_batch_size 1 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False \
--overwrite False
```

There are some important parameters:

- `encoder`: Name or path of the model to evaluate.

- `languages`: The languages you want to evaluate on. Avaliable languages: `ar de en es fr hi it ja ko pt ru th zh`.

- `cache_dir`: If you do not want to save the downloaded MLDR dataset in the default `HF_DATASET_CACHE`, you can set this parameter.

- `max_query_length` & `max_passage_length`: Maximum query length and maximum passage length when encoding.

- `batch_size` & `corpus_batch_size`: Batch size for query and corpus when encoding. If `max_query_length == max_passage_length`, you can ignore the `corpus_batch_size` parameter and only set `batch_size` for convenience. For faster evaluation, you should set the `batch_size` and `corpus_batch_size` as large as possible.

- `pooling_method` & `normalize_embeddings`: You should follow the corresponding setting of the model you are evaluating. For example, `BAAI/bge-m3` is `cls` and `True`, `intfloat/multilingual-e5-large` is `mean` and `True`, and `intfloat/e5-mistral-7b-instruct` is `last` and `True`.

- `add_instruction`: Whether to add instruction for query or passage when evaluating. If set `add_instruction=True`, you should also set the following parameters appropriately:

  - `query_instruction_for_retrieval`: the query instruction for retrieval
  - `passage_instruction_for_retrieval`: the passage instruction for retrieval

  If you only add query instruction, just ignore the `passage_instruction_for_retrieval` parameter.

- `overwrite`: Whether to overwrite evaluation results.

## Hybrid Retrieval (Dense & Sparse)

If you want to perform **hybrid retrieval with both dense and sparse methods**, you can follow the following steps:

1. Install Java, Pyserini and Faiss (CPU version or GPU version):

```bash
# install java (Linux)
apt update
apt install openjdk-11-jdk

# install pyserini
pip install pyserini

# install faiss
## CPU version
conda install -c conda-forge faiss-cpu

## GPU version
conda install -c conda-forge faiss-gpu
```

2. Download qrels from [BAAI/mldr](https://huggingface.co/datasets/BAAI/mldr/tree/main/qrels):

```bash
cd qrels
wget https://huggingface.co/datasets/BAAI/mldr/resolve/main/qrels/qrels.mldr-v1.0-*.tsv
```

3. Dense retrieval:

```bash
cd dense_retrieval

# 1. Generate Corpus Embedding
python3 step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--cache_dir /home/datasets/.cache \
--max_passage_length 8192 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 2. Search Results
python3 step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--cache_dir /home/baaiks/jianlv/datasets/.cache \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--threads 16 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 3. Print and Save Evaluation Results
python3 step2-eval_dense_mldr.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True
```

> Note: The evaluation results of this method may have slight differences compared to results of the method mentioned earlier, which is considered normal.

4. Sparse Retrieval

```bash
cd sparse_retrieval
# 1. Generate Query and Corpus Sparse Vector
python3 step0-encode_query-and-corpus.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--save_dir ./encoded_query-and-corpus \
--cache_dir /home/baaiks/jianlv/datasets/.cache \
--max_query_length 512 \
--max_passage_length 8192 \
--batch_size 1024 \
--corpus_batch_size 4 \
--pooling_method cls \
--normalize_embeddings True

# 2. Output Search Results
python3 step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--encoded_query_and_corpus_save_dir ./encoded_query-and-corpus \
--result_save_dir ./search_results \
--threads 16 \
--hits 1000

# 3. Print and Save Evaluation Results
python3 step2-eval_sparse_mldr.py \
--encoder BAAI/bge-m3 \
--languages ar de es fr hi it ja ko pt ru th en zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True
```

5. Hybrid Retrieval

```bash
cd hybrid_retrieval

# 1. Search Dense and Sparse Results
Dense Retrieval
Sparse Retrieval

# 2. Hybrid Dense and Sparse Search Results
python3 step0-hybrid_search_results.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--dense_search_result_save_dir ../dense_retrieval/search_results \
--sparse_search_result_save_dir ../sparse_retrieval/search_results \
--hybrid_result_save_dir ./search_results \
--top_k 1000 \
--dense_weight 0.2 --sparse_weight 0.8

# 3. Print and Save Evaluation Results
python3 step1-eval_hybrid_mldr.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True
```

