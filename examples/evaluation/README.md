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

### 2. BEIR



### 3. MSMARCO



### 4. MIRACL



### 5. MLDR



### 6. MKQA



### 7. AIR+Bench



### 8. Custom Dataset

