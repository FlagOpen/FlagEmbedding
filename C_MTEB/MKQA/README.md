# MKQA

MKQA is a cross-lingual question answering dataset covering 25 non-English languages. For more details, please refer to [here](https://github.com/apple/ml-mkqa).

We filter questions which types are `unanswerable`, `binary` and `long-answer`. Finally we get 6,619 questions for every language. To perform evaluation, you should firstly **download the test data**:
```bash
# download
wget https://huggingface.co/datasets/Shitao/bge-m3-data/resolve/main/MKQA_test-data.zip
# unzip to `qa_data` dir
unzip MKQA_test-data.zip -d qa_data
```

We use the well-processed NQ [corpus](https://huggingface.co/datasets/BeIR/nq) offered by BEIR as the candidate, and perform evaluation with metrics: Recall@100 and Recall@20. Here the definition of Recall@k refers to [RocketQA](https://aclanthology.org/2021.naacl-main.466.pdf).

## Dense Retrieval

If you only want to perform dense retrieval with embedding models, you can follow the following steps:

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

2. Dense retrieval:

```bash
cd dense_retrieval

# 1. Generate Corpus Embedding
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--index_save_dir ./corpus-index \
--max_passage_length 512 \
--batch_size 256 \
--fp16 \
--add_instruction False \
--pooling_method cls \
--normalize_embeddings True

# 2. Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--threads 16 \
--batch_size 32 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 3. Print and Save Evaluation Results
python step2-eval_dense_mkqa.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32 \
--pooling_method cls \
--normalize_embeddings True
```

There are some important parameters:

- `encoder`: Name or path of the model to evaluate.

- `languages`: The languages you want to evaluate on. Avaliable languages: `ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw`.

- `max_passage_length`: Maximum passage length when encoding.

- `batch_size`: Batch size for query and corpus when encoding. For faster evaluation, you should set the `batch_size` as large as possible.

- `pooling_method` & `normalize_embeddings`: You should follow the corresponding setting of the model you are evaluating. For example, `BAAI/bge-m3` is `cls` and `True`, `intfloat/multilingual-e5-large` is `mean` and `True`, and `intfloat/e5-mistral-7b-instruct` is `last` and `True`.

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

2. Dense retrieval:

```bash
cd dense_retrieval

# 1. Generate Corpus Embedding
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--index_save_dir ./corpus-index \
--max_passage_length 512 \
--batch_size 256 \
--fp16 \
--add_instruction False \
--pooling_method cls \
--normalize_embeddings True

# 2. Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--threads 16 \
--batch_size 32 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 3. Print and Save Evaluation Results
python step2-eval_dense_mkqa.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32 \
--pooling_method cls \
--normalize_embeddings True
```

3. Sparse Retrieval

```bash
cd sparse_retrieval

# 1. Generate Query and Corpus Sparse Vector
python step0-encode_query-and-corpus.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--qa_data_dir ../qa_data \
--save_dir ./encoded_query-and-corpus \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 1024 \
--pooling_method cls \
--normalize_embeddings True

# 2. Output Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--encoded_query_and_corpus_save_dir ./encoded_query-and-corpus \
--result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--threads 16 \
--hits 1000

# 3. Print and Save Evaluation Results
python step2-eval_sparse_mkqa.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32 \
--pooling_method cls \
--normalize_embeddings True
```

4. Hybrid Retrieval

```bash
cd hybrid_retrieval

# 1. Search Dense and Sparse Results
Dense Retrieval
Sparse Retrieval

# 2. Hybrid Dense and Sparse Search Results
python step0-hybrid_search_results.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--dense_search_result_save_dir ../dense_retrieval/search_results \
--sparse_search_result_save_dir ../sparse_retrieval/search_results \
--hybrid_result_save_dir ./search_results \
--top_k 1000 \
--dense_weight 1 --sparse_weight 0.3 \
--threads 32

# 3. Print and Save Evaluation Results
python step1-eval_hybrid_mkqa.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw  \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32 \
--pooling_method cls \
--normalize_embeddings True
```

## MultiVector and All Rerank

If you want to perform **multi-vector reranking** or **all reranking** based on the search results of dense retrieval, you can follow the following steps:

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

2. Dense retrieval:

```bash
cd dense_retrieval

# 1. Generate Corpus Embedding
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--index_save_dir ./corpus-index \
--max_passage_length 512 \
--batch_size 256 \
--fp16 \
--add_instruction False \
--pooling_method cls \
--normalize_embeddings True

# 2. Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--threads 16 \
--batch_size 32 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 3. Print and Save Evaluation Results
python step2-eval_dense_mkqa.py \
--encoder BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32 \
--pooling_method cls \
--normalize_embeddings True
```

3. Rerank search results with multi-vector scores or all scores: 

```bash
cd multi_vector_rerank

# 1. Rerank Search Results
python step0-rerank_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ../dense_retrieval/search_results \
--qa_data_dir ../qa_data \
--rerank_result_save_dir ./rerank_results \
--top_k 100 \
--batch_size 4 \
--max_length 512 \
--pooling_method cls \
--normalize_embeddings True \
--dense_weight 1 --sparse_weight 0.3 --colbert_weight 1 \
--num_shards 1 --shard_id 0 --cuda_id 0

# 2. Print and Save Evaluation Results
python step1-eval_rerank_mkqa.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./rerank_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32
```

>**Note**: 
>
>- You should set `dense_weight`, `sparse_weight` and `colbert_weight` based on the downstream task scenario. If the dense method performs well while the sparse method does not, you can lower `sparse_weight` and increase `dense_weight` accordingly.
>
>- Based on our experience, dividing the sentence pairs to be reranked into several shards and computing scores for each shard on a single GPU tends to be more efficient than using multiple GPUs to compute scores for all sentence pairs directly.Therefore, if your machine have multiple GPUs, you can set `num_shards` to the number of GPUs and launch multiple terminals to execute the command (`shard_id` should be equal to `cuda_id`). Therefore, if you have multiple GPUs on your machine, you can launch multiple terminals and run multiple commands simultaneously. Make sure to set the `shard_id` and `cuda_id` appropriately, and ensure that you have computed scores for all shards before proceeding to the second step.

4. (*Optional*) In the 3rd step, you can get all three kinds of scores, saved to `rerank_result_save_dir/dense/{encoder}-{reranker}`, `rerank_result_save_dir/sparse/{encoder}-{reranker}` and `rerank_result_save_dir/colbert/{encoder}-{reranker}`. If you want to try other weights, you don't need to rerun the 4th step. Instead, you can use [this script](./multi_vector_rerank/hybrid_all_results.py) to hybrid the three kinds of scores directly.

```bash
cd multi_vector_rerank

# 1. Hybrid All Search Results
python hybrid_all_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--dense_search_result_save_dir ./rerank_results/dense \
--sparse_search_result_save_dir ./rerank_results/sparse \
--colbert_search_result_save_dir ./rerank_results/colbert \
--hybrid_result_save_dir ./hybrid_search_results \
--top_k 200 \
--threads 32 \
--dense_weight 1 --sparse_weight 0.1 --colbert_weight 1

# 2. Print and Save Evaluation Results
python step1-eval_rerank_mkqa.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./hybrid_search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_hybrid_results \
--metrics recall@20 recall@100 \
--threads 32
```


## BM25 Baseline

We provide two methods of evaluating BM25 baseline:

1. Use the same tokenizer with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (i.e., tokenizer of [XLM-Roberta](https://huggingface.co/FacebookAI/xlm-roberta-large)):

```bash
cd sparse_retrieval

# 1. Output Search Results with BM25
python bm25_baseline_same_tokenizer.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mkqa.py \
--encoder bm25_same_tokenizer \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32
```

2. Use the language analyzer provided by [Anserini](https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/analysis/AnalyzerMap.java) ([Lucene Tokenizer](https://github.com/apache/lucene/tree/main/lucene/analysis/common/src/java/org/apache/lucene/analysis)):

```bash
cd sparse_retrieval

# 1. Output Search Results with BM25
python bm25_baseline.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mkqa.py \
--encoder bm25 \
--languages ar da de es fi fr he hu it ja km ko ms nl no pl pt ru sv th tr vi zh_cn zh_hk zh_tw \
--search_result_save_dir ./search_results \
--qa_data_dir ../qa_data \
--eval_result_save_dir ./eval_results \
--metrics recall@20 recall@100 \
--threads 32
```

