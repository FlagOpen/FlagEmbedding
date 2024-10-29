# Evaluation

After finetuning, the model needs to be evaluated. To facilitate this, we have provided scripts for assessing it on various datasets, including **MTEB**, **BEIR**, **MSMARCO**, **MIRACL**, **MLDR**, **MKQA**, and **AIR-Bench**. You can find the specific bash scripts in the respective folders. This document provides an overview of these evaluations.

First, we will introduce the commonly used variables, followed by an introduction to the variables for each dataset.

## Introduction

### 1. EvalArgs

**Parameters for evaluation setup:**

- **`eval_name`**: Name of the evaluation task (e.g., msmarco, beir, miracl).
  
- **`dataset_dir`**: Path to the dataset directory. This can be:
    1. A local path to perform evaluation on your dataset (must exist). It should contain:
        - `corpus.jsonl`
        - `<split>_queries.jsonl`
        - `<split>_qrels.jsonl`
    2. Path to store datasets downloaded via API. Provide `None` to use the cache directory.
  
- **`force_redownload`**: Set to `true` to force redownload of the dataset.

- **`dataset_names`**: List of dataset names to evaluate or `None` to evaluate all available datasets.

- **`splits`**: Dataset splits to evaluate. Default is `test`.

- **`corpus_embd_save_dir`**: Directory to save corpus embeddings. If `None`, embeddings will not be saved.

- **`output_dir`**: Directory to save evaluation results.

- **`search_top_k`**: Top-K results for initial retrieval.

- **`rerank_top_k`**: Top-K results for reranking.

- **`cache_path`**: Cache directory for datasets.

- **`token`**: Token used for accessing the model.

- **`overwrite`**: Set to `true` to overwrite existing evaluation results.

- **`ignore_identical_ids`**: Set to `true` to ignore identical IDs in search results.

- **`k_values`**: List of K values for evaluation (e.g., [1, 3, 5, 10, 100, 1000]).

- **`eval_output_method`**: Format for outputting evaluation results (options: 'json', 'markdown'). Default is `markdown`.

- **`eval_output_path`**: Path to save the evaluation output.

- **`eval_metrics`**: Metrics used for evaluation (e.g., ['ndcg_at_10', 'recall_at_10']).

### 2. ModelArgs

**Parameters for Model Configuration:**

- **`embedder_name_or_path`**: The name or path to the embedder.

- **`embedder_model_class`**: Class of the model used for embedding (options include 'auto', 'encoder-only-base', etc.). Default is `auto`.

- **`normalize_embeddings`**: Set to `true` to normalize embeddings.

- **`use_fp16`**: Use FP16 precision for inference.

- **`devices`**: List of devices used for inference.

- **`query_instruction_for_retrieval`**, **`query_instruction_format_for_retrieval`**: Instructions and format for query during retrieval.

- **`examples_for_task`**, **`examples_instruction_format`**: Example tasks and their instructions format.

- **`trust_remote_code`**: Set to `true` to trust remote code execution.

- **`reranker_name_or_path`**: Name or path to the reranker.

- **`reranker_model_class`**: Reranker model class (options include 'auto', 'decoder-only-base', etc.). Default is `auto`.

- **`reranker_peft_path`**: Path for portable encoder fine-tuning of the reranker.

- **`use_bf16`**: Use BF16 precision for inference.

- **`query_instruction_for_rerank`**, **`query_instruction_format_for_rerank`**: Instructions and format for query during reranking.

- **`passage_instruction_for_rerank`**, **`passage_instruction_format_for_rerank`**: Instructions and format for processing passages during reranking.

- **`cache_dir`**: Cache directory for models.

- **`embedder_batch_size`**, **`reranker_batch_size`**: Batch sizes for embedding and reranking.

- **`embedder_query_max_length`**, **`embedder_passage_max_length`**: Maximum length for embedding queries and passages.

- **`reranker_query_max_length`**, **`reranker_max_length`**: Maximum lengths for reranking queries and reranking in general.

- **`normalize`**: Normalize the reranking scores.

- **`prompt`**: Prompt for the reranker.

- **`cutoff_layers`**, **`compress_ratio`**, **`compress_layers`**: Parameters for configuring the output and compression of layerwise or lightweight rerankers.

## Usage

### 1. MTEB

In the evaluation of MTEB, we primarily utilize the official [MTEB](https://github.com/embeddings-benchmark/mteb) code, which supports only the assessment of embedders. Additionally, it restricts the output format of evaluation results to JSON. The following new variables have been introduced:

- **`languages`**: Languages to evaluate. Default: eng
- **`tasks`**: Tasks to evaluate. Default: None
- **`task_types`**: The task types to evaluate. Default: None
- **`use_special_instructions`**: Whether to use specific instructions in `prompts.py` for evaluation. Default: False
- **`use_special_examples`**: Whether to use specific examples in `examples.py` for evaluation. Default: False

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.mteb \
	--eval_name mteb \
    --output_dir ./data/mteb/search_results \
    --languages eng \
    --tasks NFCorpus BiorxivClusteringS2S SciDocsRR \
    --eval_output_path ./mteb/mteb_eval_results.json \
    --embedder_name_or_path BAAI/bge-m3 \
    --devices cuda:7 \
    --cache_dir ./cache/model
```

### 2. BEIR

[BEIR](https://github.com/beir-cellar/beir/) supports evaluations on datasets including `arguana`, `climate-fever`, `cqadupstack`, `dbpedia-entity`, `fever`, `fiqa`, `hotpotqa`, `msmarco`, `nfcorpus`, `nq`, `quora`, `scidocs`, `scifact`, `trec-covid`, `webis-touche2020`, with `msmarco` as the dev set and all others as test sets. The following new variables have been introduced:

- **`use_special_instructions`**: Whether to use specific instructions in `prompts.py` for evaluation. Default: False

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.beir \
	--eval_name beir \
    --dataset_dir ./beir/data \
    --dataset_names fiqa arguana cqadupstack \
    --splits test dev \
    --corpus_embd_save_dir ./beir/corpus_embd \
    --output_dir ./beir/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./beir/beir_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```

### 3. MSMARCO

[MSMARCO](https://microsoft.github.io/msmarco/) supports evaluations on both `passage` and `document`, providing evaluation splits for `dev`, `dl19`, and `dl20` respectively.

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.msmarco \
	--eval_name msmarco \
    --dataset_dir ./msmarco/data \
    --dataset_names passage \
    --splits dev dl19 dl20 \
    --corpus_embd_save_dir ./msmarco/corpus_embd \
    --output_dir ./msmarco/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite True \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./msmarco/msmarco_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```

### 4. MIRACL

[MIRACL](https://github.com/project-miracl/miracl) supports evaluations in multiple languages. We utilize different languages as dataset names, including `ar`, `bn`, `en`, `es`, `fa`, `fi`, `fr`, `hi`, `id`, `ja`, `ko`, `ru`, `sw`, `te`, `th`, `zh`, `de`, `yo`. For the languages `de` and `yo`, the supported splits are `dev`, while for the rest, the supported splits are `train` and `dev`.

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.miracl \
	--eval_name miracl \
    --dataset_dir ./miracl/data \
    --dataset_names bn hi sw te th yo \
    --splits dev \
    --corpus_embd_save_dir ./miracl/corpus_embd \
    --output_dir ./miracl/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./miracl/miracl_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```

### 5. MLDR

[MLDR](https://huggingface.co/datasets/Shitao/MLDR) supports evaluations in multiple languages. We have dataset names in various languages, including `ar`, `de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `pt`, `ru`, `th`, `zh`. The available splits are `train`, `dev`, and `test`.

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.mldr \
	--eval_name mldr \
    --dataset_dir ./mldr/data \
    --dataset_names hi \
    --splits test \
    --corpus_embd_save_dir ./mldr/corpus_embd \
    --output_dir ./mldr/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./mldr/mldr_eval_results.md \
    --eval_metrics ndcg_at_10 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```

### 6. MKQA

[MKQA](https://aclanthology.org/2021.tacl-1.82/) supports multi-language evaluation, using different languages as dataset names, including `en`, `ar`, `fi`, `ja`, `ko`, `ru`, `es`, `sv`, `he`, `th`, `da`, `de`, `fr`, `it`, `nl`, `pl`, `pt`, `hu`, `vi`, `ms`, `km`, `no`, `tr`, `zh_cn`, `zh_hk`, `zh_tw`. The supported split is `test`.

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.mkqa \
	--eval_name mkqa \
    --dataset_dir ./mkqa/data \
    --dataset_names en zh_cn \
    --splits test \
    --corpus_embd_save_dir ./mkqa/corpus_embd \
    --output_dir ./mkqa/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite False \
    --k_values 20 \
    --eval_output_method markdown \
    --eval_output_path ./mkqa/mkqa_eval_results.md \
    --eval_metrics qa_recall_at_20 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```

### 7. AIR-Bench

The AIR-Bench is mainly based on the official [AIR-Bench](https://github.com/AIR-Bench/AIR-Bench/tree/main) framework, and it necessitates the use of official evaluation metrics. Below are some important variables:

- **`benchmark_version`**: Benchmark version.
- **`task_types`**: Task types.
- **`domains`**: Domains to evaluate.
- **`languages`**: Languages to evaluate.

Here is an example for evaluation:

```shell
python -m FlagEmbedding.evaluation.air_bench \
	--benchmark_version AIR-Bench_24.05 \
    --task_types qa long-doc \
    --domains arxiv \
    --languages en \
    --splits dev test \
    --output_dir ./air_bench/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_dir ./cache/data \
    --overwrite False \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --model_cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```

### 8. Custom Dataset

You can refer to [MLDR dataset](https://github.com/hanhainebula/FlagEmbedding/tree/new-flagembedding-v1/FlagEmbedding/evaluation/mldr), just need to rewrite `DataLoader`, rewriting the loading method for the required dataset.

The example data for `corpus.jsonl`:

```json
{"id": "77628", "title": "Recover deleted cache", "text": "Is it possible to recover cache photos? The files were deleted by Clean Master to save space. I have no idea where to start. The photos are precious and are irreplaceable."}
{"id": "806", "title": "How do I undelete or recover deleted files on Android?", "text": "> **Possible Duplicate:**   >  How can I recover a deleted file on Android? Is there a way to recover deleted files on Android phones without using standard USB storage recovery tools?"}
{"id": "74923", "title": "Recovering deleted pictures", "text": "I recently deleted all of my pictures by mistake from my samsung galaxy s4.   I went into my files and documents and deleted not realising it would delete all my pics! Is there a way for me to recover them? My phone is not rooted. I have not taken any pictures since but have received pictures through whatsapp?"}
{"id": "50864", "title": "How to recover deleted files on Android phone", "text": "I was a using an autocall recorder app on my HTC Wildfire. I saved a call on my phones SD card and in my Dropbox. However, I accidently deleted the saved call and it was removed from my dropbox file. I now need this call and I tried some data recovery software. I scanned both my phone and pc. The software found the deleted call and recovered it, but the file which has .AMR extension does not work. The size of the file is only 143kb.   1. What is the likelihood this file is corrupted/stiil intact? Can I check that?   2. Which software can I use to salvage/replay the AMR file?"}
{"id": "81285", "title": "How to recover deleted photo album saved on internal memory - Note 3", "text": "I have a Samsung Note 3 and I accidentally deleted an entire photo album from my phones gallery. I didn't enable my device to sync with Gmail. I didn't manually backup any of the data. The images were saved on my phone, not on the SD card. Is there any way for me to recover this deleted photo album? I Google'd and came across SDrescan but that won't work since the images were not initially saved on my SD card."}
```

The example data for `test_queries.jsonl`:

```json
{"id": "79085", "text": "HTC One Mini data recovery after root"}
```

The example data for `test_qrels.jsonl`:

```json
{"qid": "79085", "docid": "77628", "relevance": 1}
{"qid": "79085", "docid": "806", "relevance": 1}
{"qid": "79085", "docid": "74923", "relevance": 1}
{"qid": "79085", "docid": "50864", "relevance": 1}
{"qid": "79085", "docid": "81285", "relevance": 1}
```

