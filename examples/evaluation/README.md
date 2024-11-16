# Evaluation

After fine-tuning the model, it is essential to evaluate its performance. To facilitate this process, we have provided scripts for assessing the model on various datasets. These datasets include: [**MTEB**](https://github.com/embeddings-benchmark/mteb), [**BEIR**](https://github.com/beir-cellar/beir), [**MSMARCO**](https://microsoft.github.io/msmarco/), [**MIRACL**](https://github.com/project-miracl/miracl), [**MLDR**](https://huggingface.co/datasets/Shitao/MLDR), [**MKQA**](https://github.com/apple/ml-mkqa), [**AIR-Bench**](https://github.com/AIR-Bench/AIR-Bench), and your **custom datasets**.

To evaluate the model on a specific dataset, you can find the corresponding bash scripts in the respective folders dedicated to each dataset. These scripts contain the necessary commands and configurations to run the evaluation process.

This document serves as an overview of the evaluation process and provides a brief introduction to each dataset.

In this section, we will first introduce the commonly used arguments across all datasets. Then, we will provide a more detailed explanation of the specific arguments used for each individual dataset.

- [1. Introduction](#1-Introduction)
  - [(1) EvalArgs](#1-EvalArgs)
  - [(2) ModelArgs](#2-ModelArgs)
- [2. Usage](#2-Usage)
  - [Requirements](#Requirements)
  - [(1) MTEB](#1-MTEB)
  - [(2) BEIR](#2-BEIR)
  - [(3) MSMARCO](#3-MSMARCO)
  - [(4) MIRACL](#4-MIRACL)
  - [(5) MLDR](#5-MLDR)
  - [(6) MKQA](#6-MKQA)
  - [(7) AIR-Bench](#7-Air-Bench)
  - [(8) Custom Dataset](#8-Custom-Dataset)

## Introduction

### 1. EvalArgs

**Arguments for evaluation setup:**

- **`eval_name`**: Name of the evaluation task (e.g., msmarco, beir, miracl).

- **`dataset_dir`**: Path to the dataset directory. This can be:
  1. A local path to perform evaluation on your dataset (must exist). It should contain:
     - `corpus.jsonl`
     - `<split>_queries.jsonl`
     - `<split>_qrels.jsonl`
  2. Path to store datasets downloaded via API. Provide `None` to use the cache directory.

- **`force_redownload`**: Set to `True` to force redownload of the dataset. Default is `False`.

- **`dataset_names`**: List of dataset names to evaluate or `None` to evaluate all available datasets. This can be the dataset name (BEIR, etc.) or language (MIRACL, etc.).

- **`splits`**: Dataset splits to evaluate. Default is `test`.

- **`corpus_embd_save_dir`**: Directory to save corpus embeddings. If `None`, embeddings will not be saved.

- **`output_dir`**: Directory to save evaluation results.

- **`search_top_k`**: Top-K results for initial retrieval. Default is `1000`.

- **`rerank_top_k`**: Top-K results for reranking. Default is `100`.

- **`cache_path`**: Cache directory for datasets. Default is `None`.

- **`token`**: Token used for accessing the private data (datasets/models) in HF. Default is `None`, which means it will use the environment variable `HF_TOKEN`.

- **`overwrite`**: Set to `True` to overwrite existing evaluation results. Default is `False`.

- **`ignore_identical_ids`**: Set to `True` to ignore identical IDs in search results. Default is `False`.

- **`k_values`**: List of K values for evaluation (e.g., [1, 3, 5, 10, 100, 1000]). Default is `[1, 3, 5, 10, 100, 1000]`.

- **`eval_output_method`**: Format for outputting evaluation results (options: 'json', 'markdown'). Default is `markdown`.

- **`eval_output_path`**: Path to save the evaluation output.

- **`eval_metrics`**: Metrics used for evaluation (e.g., ['ndcg_at_10', 'recall_at_10']). Default is `[ndcg_at_10, recall_at_100]`.

### 2. ModelArgs

**Arguments for Model Configuration:**

- **`embedder_name_or_path`**: The name or path to the embedder.
- **`embedder_model_class`**: Class of the model used for embedding (current options include 'encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl'.). Default is None. For the custom model, you should set this argument.
- **`normalize_embeddings`**: Set to `True` to normalize embeddings.
- **`pooling_method`**: The pooling method for the embedder.
- **`use_fp16`**: Use FP16 precision for inference.
- **`devices`**: List of devices used for inference.
- **`query_instruction_for_retrieval`**, **`query_instruction_format_for_retrieval`**: Instructions and format for query during retrieval.
- **`examples_for_task`**, **`examples_instruction_format`**: Example tasks and their instructions format.
- **`trust_remote_code`**: Set to `True` to trust remote code execution.
- **`reranker_name_or_path`**: Name or path to the reranker.
- **`reranker_model_class`**: Reranker model class (options include 'encoder-only-base', 'decoder-only-base', 'decoder-only-layerwise', 'decoder-only-lightweight'). Default is None. For the custom model, you should set this argument.
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
- **`cutoff_layers`**, **`compress_ratio`**, **`compress_layers`**: arguments for configuring the output and compression of layerwise or lightweight rerankers.

***Notice:*** If you evaluate your own model, please set `embedder_model_class` and `reranker_model_class`.

## Usage

### Requirements

You need install `pytrec_eval` and `faiss` for evaluation:

```shell
pip install pytrec_eval
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 1. MTEB

For MTEB, we primarily use the official [MTEB](https://github.com/embeddings-benchmark/mteb) code, which only supports the assessment of embedders. Moreover, it restricts the output format of the evaluation results to JSON. We have introduced the following new arguments:

- **`languages`**: Languages to evaluate. Default: eng
- **`tasks`**: Tasks to evaluate. Default: None
- **`task_types`**: The task types to evaluate. Default: None
- **`use_special_instructions`**: Whether to use specific instructions in `prompts.py` for evaluation. Default: False
- **`examples_path`**: Use specific examples in the path. Default: None

Here is an example for evaluation:

```shell
pip install mteb==1.15.0
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

[BEIR](https://github.com/beir-cellar/beir/) supports evaluations on datasets including `arguana`, `climate-fever`, `cqadupstack`, `dbpedia-entity`, `fever`, `fiqa`, `hotpotqa`, `msmarco`, `nfcorpus`, `nq`, `quora`, `scidocs`, `scifact`, `trec-covid`, `webis-touche2020`, with `msmarco` as the dev set and all others as test sets. The following new arguments have been introduced:

- **`use_special_instructions`**: Whether to use specific instructions in `prompts.py` for evaluation. Default: False

Here is an example for evaluation:

```shell
pip install beir
mkdir eval_beir
cd eavl_beir
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
    --ignore_identical_ids True \
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

[MKQA](https://github.com/apple/ml-mkqa) supports cross-lingual retrieval evaluation (from the [paper of BGE-M3](https://arxiv.org/pdf/2402.03216)), using different languages as dataset names, including `en`, `ar`, `fi`, `ja`, `ko`, `ru`, `es`, `sv`, `he`, `th`, `da`, `de`, `fr`, `it`, `nl`, `pl`, `pt`, `hu`, `vi`, `ms`, `km`, `no`, `tr`, `zh_cn`, `zh_hk`, `zh_tw`. The supported split is `test`.

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

The AIR-Bench is primarily based on the official [AIR-Bench repository](https://github.com/AIR-Bench/AIR-Bench/tree/main) and requires the use of its official evaluation codes. Below are some important arguments:

- **`benchmark_version`**: Benchmark version.
- **`task_types`**: Task types.
- **`domains`**: Domains to evaluate.
- **`languages`**: Languages to evaluate.

Here is an example for evaluation:

```shell
pip install air-benchmark
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

The example data for `corpus.jsonl`:

```json
{"id": "566392", "title": "", "text": "Have the check reissued to the proper payee."}
{"id": "65404", "title": "", "text": "Just have the associate sign the back and then deposit it.  It's called a third party cheque and is perfectly legal.  I wouldn't be surprised if it has a longer hold period and, as always, you don't get the money if the cheque doesn't clear. Now, you may have problems if it's a large amount or you're not very well known at the bank.  In that case you can have the associate go to the bank and endorse it in front of the teller with some ID.  You don't even technically have to be there.  Anybody can deposit money to your account if they have the account number. He could also just deposit it in his account and write a cheque to the business."}
{"id": "325273", "title": "", "text": "Sure you can.  You can fill in whatever you want in the From section of a money order, so your business name and address would be fine. The price only includes the money order itself.  You can hand deliver it yourself if you want, but if you want to mail it, you'll have to provide an envelope and a stamp. Note that, since you won't have a bank record of this payment, you'll want to make sure you keep other records, such as the stub of the money order.  You should probably also ask the contractor to give you a receipt."}
{"id": "88124", "title": "", "text": "You're confusing a lot of things here. Company B LLC will have it's sales run under Company A LLC, and cease operating as a separate entity These two are contradicting each other. If B LLC ceases to exist - it is not going to have it's sales run under A LLC, since there will be no sales to run for a non-existent company. What happens is that you merge B LLC into A LLC, and then convert A LLC into S Corp. So you're cancelling the EIN for B LLC, you're cancelling the EIN for A LLC - because both entities cease to exist. You then create a EIN for A Corp, which is the converted A LLC, and you create a DBA where A Corp DBA B Shop. You then go to the bank and open the account for A Corp DBA B Shop with the EIN you just created for A Corp. Get a better accountant. Before you convert to S-Corp."}
{"id": "285255", "title": "", "text": "\"I'm afraid the great myth of limited liability companies is that all such vehicles have instant access to credit.  Limited liability on a company with few physical assets to underwrite the loan, or with insufficient revenue, will usually mean that the owners (or others) will be asked to stand surety on any credit. However, there is a particular form of \"\"credit\"\" available to businesses on terms with their clients.  It is called factoring. Factoring is a financial transaction   whereby a business sells its accounts   receivable (i.e., invoices) to a third   party (called a factor) at a discount   in exchange for immediate money with   which to finance continued business.   Factoring differs from a bank loan in   three main ways. First, the emphasis   is on the value of the receivables   (essentially a financial asset), not   the firm’s credit worthiness.   Secondly, factoring is not a loan – it   is the purchase of a financial asset   (the receivable). Finally, a bank loan   involves two parties whereas factoring   involves three. Recognise that this can be quite expensive.  Most banks catering to small businesses will offer some form of factoring service, or will know of services that offer it.  It isn't that different from cheque encashment services (pay-day services) where you offer a discount on future income for money now. An alternative is simply to ask his clients if they'll pay him faster if he offers a discount (since either of interest payments or factoring would reduce profitability anyway).\""}
{"id": "350819", "title": "", "text": "Banks will usually look at 2 years worth of tax returns for issuing business credit.  If those aren't available (for instance, for recently formed businesses), they will look at the personal returns of the owners. Unfortunately, it sounds like your friend is in the latter category. Bringing in another partner isn't necessarily going to help, either; with only two partners / owners, the bank would probably look at both owners' personal tax returns and credit histories.  It may be necessary to offer collateral. I'm sorry I can't offer any better solutions, but alternative funding such as personal loans from family & friends could be necessary.  Perhaps making them partners in exchange for capital."}
```

The example data for `test_queries.jsonl`:

```json
{"id": "8", "text": "How to deposit a cheque issued to an associate in my business into my business account?"}
{"id": "15", "text": "Can I send a money order from USPS as a business?"}
{"id": "18", "text": "1 EIN doing business under multiple business names"}
{"id": "26", "text": "Applying for and receiving business credit"}
```

The example data for `test_qrels.jsonl`:

```json
{"qid": "8", "docid": "566392", "relevance": 1}
{"qid": "8", "docid": "65404", "relevance": 1}
{"qid": "15", "docid": "325273", "relevance": 1}
{"qid": "18", "docid": "88124", "relevance": 1}
{"qid": "26", "docid": "285255", "relevance": 1}
{"qid": "26", "docid": "350819", "relevance": 1}
```

Please put the above file (`corpus.jsonl`, `test_queries.jsonl`, `test_qrels.jsonl`) in `dataset_dir`, and then you can use the following code:

```shell
python -m FlagEmbedding.evaluation.custom \
    --eval_name your_data_name \
    --dataset_dir ./your_data_path \
    --splits test \
    --corpus_embd_save_dir ./your_data_name/corpus_embd \
    --output_dir ./your_data_name/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path ./cache/data \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./your_data_name/eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir ./cache/model \
    --reranker_query_max_length 512 \
    --reranker_max_length 1024 
```