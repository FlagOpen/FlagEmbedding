# MultiLongDocRetrieval

MultiLongDocRetrieval (denoted as MLDR) is a multilingual long-document retrieval dataset. For more details, please refer to [Shitao/MLDR](https://huggingface.co/datasets/Shitao/MLDR).

## Dense Retrieval

This task has been merged into [MTEB](https://github.com/embeddings-benchmark/mteb), you can easily use mteb tool to do evaluation.   

We also provide a [script](./mteb_dense_eval/eval_MLDR.py), you can use it following this command:

```bash
cd mteb_dense_eval

# Print and Save Evaluation Results with MTEB
python eval_MLDR.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--results_save_path ./results \
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

2. Download qrels from [Shitao/MLDR](https://huggingface.co/datasets/Shitao/MLDR/tree/main/qrels):

```bash
mkdir -p qrels
cd qrels

splits=(dev test)
langs=(ar de en es fr hi it ja ko pt ru th zh)
for split in ${splits[*]}; do for lang in ${langs[*]}; do wget "https://huggingface.co/datasets/Shitao/MLDR/resolve/main/qrels/qrels.mldr-v1.0-${lang}-${split}.tsv"; done; done;
```

3. Dense retrieval:

```bash
cd dense_retrieval

# 1. Generate Corpus Embedding
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--max_passage_length 8192 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 2. Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--threads 16 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 3. Print and Save Evaluation Results
python step2-eval_dense_mldr.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True
```

> Note: The evaluation results of this method may have slight differences compared to results of the method mentioned earlier (*with MTEB*), which is considered normal.

4. Sparse Retrieval

```bash
cd sparse_retrieval

# 1. Generate Query and Corpus Sparse Vector
python step0-encode_query-and-corpus.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--save_dir ./encoded_query-and-corpus \
--max_query_length 512 \
--max_passage_length 8192 \
--batch_size 1024 \
--corpus_batch_size 4 \
--pooling_method cls \
--normalize_embeddings True

# 2. Output Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--encoded_query_and_corpus_save_dir ./encoded_query-and-corpus \
--result_save_dir ./search_results \
--threads 16 \
--hits 1000

# 3. Print and Save Evaluation Results
python step2-eval_sparse_mldr.py \
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
python step0-hybrid_search_results.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--dense_search_result_save_dir ../dense_retrieval/search_results \
--sparse_search_result_save_dir ../sparse_retrieval/search_results \
--hybrid_result_save_dir ./search_results \
--top_k 1000 \
--dense_weight 0.2 --sparse_weight 0.8

# 3. Print and Save Evaluation Results
python step1-eval_hybrid_mldr.py \
--model_name_or_path BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
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

2. Download qrels from [Shitao/MLDR](https://huggingface.co/datasets/Shitao/MLDR/tree/main/qrels):

```bash
mkdir -p qrels
cd qrels

splits=(dev test)
langs=(ar de en es fr hi it ja ko pt ru th zh)
for split in ${splits[*]}; do for lang in ${langs[*]}; do wget "https://huggingface.co/datasets/Shitao/MLDR/resolve/main/qrels/qrels.mldr-v1.0-${lang}-${split}.tsv"; done; done;
```

3. Dense retrieval:

```bash
cd dense_retrieval

# 1. Generate Corpus Embedding
python step0-generate_embedding.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--max_passage_length 8192 \
--batch_size 4 \
--fp16 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 2. Search Results
python step1-search_results.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--index_save_dir ./corpus-index \
--result_save_dir ./search_results \
--threads 16 \
--hits 1000 \
--pooling_method cls \
--normalize_embeddings True \
--add_instruction False

# 3. Print and Save Evaluation Results
python step2-eval_dense_mldr.py \
--encoder BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10 \
--pooling_method cls \
--normalize_embeddings True
```

> **Note**: The evaluation results of this method may have slight differences compared to results of the method mentioned earlier (*with MTEB*), which is considered normal.

4. Rerank search results with multi-vector scores or all scores: 

```bash
cd multi_vector_rerank

# 1. Rerank Search Results
python step0-rerank_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ../dense_retrieval/search_results \
--rerank_result_save_dir ./rerank_results \
--top_k 200 \
--batch_size 4 \
--max_query_length 512 \
--max_passage_length 8192 \
--pooling_method cls \
--normalize_embeddings True \
--dense_weight 0.15 --sparse_weight 0.5 --colbert_weight 0.35 \
--num_shards 1 --shard_id 0 --cuda_id 0

# 2. Print and Save Evaluation Results
python step1-eval_rerank_mldr.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./rerank_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10
```

>**Note**: 
>
>- You should set `dense_weight`, `sparse_weight` and `colbert_weight` based on the downstream task scenario. If the dense method performs well while the sparse method does not, you can lower `sparse_weight` and increase `dense_weight` accordingly.
>
>- Based on our experience, dividing the sentence pairs to be reranked into several shards and computing scores for each shard on a single GPU tends to be more efficient than using multiple GPUs to compute scores for all sentence pairs directly.Therefore, if your machine have multiple GPUs, you can set `num_shards` to the number of GPUs and launch multiple terminals to execute the command (`shard_id` should be equal to `cuda_id`). Therefore, if you have multiple GPUs on your machine, you can launch multiple terminals and run multiple commands simultaneously. Make sure to set the `shard_id` and `cuda_id` appropriately, and ensure that you have computed scores for all shards before proceeding to the second step.

5. (*Optional*) In the 4th step, you can get all three kinds of scores, saved to `rerank_result_save_dir/dense/{encoder}-{reranker}`, `rerank_result_save_dir/sparse/{encoder}-{reranker}` and `rerank_result_save_dir/colbert/{encoder}-{reranker}`. If you want to try other weights, you don't need to rerun the 4th step. Instead, you can use [this script](./multi_vector_rerank/hybrid_all_results.py) to hybrid the three kinds of scores directly.

```bash
cd multi_vector_rerank

# 1. Hybrid All Search Results
python hybrid_all_results.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--dense_search_result_save_dir ./rerank_results/dense \
--sparse_search_result_save_dir ./rerank_results/sparse \
--colbert_search_result_save_dir ./rerank_results/colbert \
--hybrid_result_save_dir ./hybrid_search_results \
--top_k 200 \
--dense_weight 0.2 --sparse_weight 0.4 --colbert_weight 0.4

# 2. Print and Save Evaluation Results
python step1-eval_rerank_mldr.py \
--encoder BAAI/bge-m3 \
--reranker BAAI/bge-m3 \
--languages ar de en es fr hi it ja ko pt ru th zh \
--search_result_save_dir ./hybrid_search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_hybrid_results \
--metrics ndcg@10
```

## BM25 Baseline

We provide two methods of evaluating BM25 baseline:

1. Use the same tokenizer with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (i.e., tokenizer of [XLM-Roberta](https://huggingface.co/FacebookAI/xlm-roberta-large)):

```bash
cd sparse_retrieval

# 1. Output Search Results with BM25 (same)
python bm25_baseline_same_tokenizer.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mldr.py \
--encoder bm25_same_tokenizer \
--languages ar de es fr hi it ja ko pt ru th en zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10
```

2. Use the language analyzer provided by [Anserini](https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/analysis/AnalyzerMap.java) ([Lucene Tokenizer](https://github.com/apache/lucene/tree/main/lucene/analysis/common/src/java/org/apache/lucene/analysis)):

```bash
cd sparse_retrieval

# 1. Output Search Results with BM25
python bm25_baseline.py

# 2. Print and Save Evaluation Results
python step2-eval_sparse_mldr.py \
--encoder bm25 \
--languages ar de es fr hi it ja ko pt ru th en zh \
--search_result_save_dir ./search_results \
--qrels_dir ../qrels \
--eval_result_save_dir ./eval_results \
--metrics ndcg@10
```



