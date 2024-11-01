# 1. Introduction

In this example, we show how to use scripts to make your fine-tuning process more convenient

# 2. Installation

```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding/scripts
```

# 3. Usage

### Hard Negatives

Hard negatives is a widely used method to improve the quality of sentence embedding. You can mine hard negatives following this command:

```shell
python hn_mine.py \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--similarity_range 0.3-0.8 \
--negative_number 15 \
--use_gpu_for_searching 
```

- **`input_file`**: json data for finetuning. This script will retrieve top-k documents for each query, and random sample negatives from the top-k documents (not including the positive documents).
- **`output_file`**: path to save JSON data with mined hard negatives for finetuning
- **`negative_number`**: the number of sampled negatives
- **`range_for_sampling`**: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)**
- **`similarity_range`**: Specifies the similarity score range for sampling negatives. This defines the range of similarity between the query and the negative samples. For example, "0.3-0.8" will only sample negatives with similarity scores between 0.3 and 0.8, allowing control over the difficulty of the negatives based on their relevance to the query. (e.g., setting it to "0.1-0.9" to sample negatives with similarity scores from 0.1 to 0.9), you can reduce the difficulty by including more diverse and less relevant negatives, whereas narrowing the range (e.g., "0.6-0.8") increases difficulty by focusing on more relevant negatives.
- **`candidate_pool`**: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. The format of this file is the same as [pretrain data](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain#2-data-format). If input a candidate_pool, this script will retrieve negatives from this file.
- **`use_gpu_for_searching`**: whether to use faiss-gpu to retrieve negatives.

### Teacher Scores

Teacher scores can be used for model distillation. You can obtain the scores using the following command:

```shell
python add_reranker_score.py \
--input_file toy_finetune_data_minedHN.jsonl \
--output_file toy_finetune_data_score.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- **`input_file`**: path to save JSON data with mined hard negatives for finetuning
- **`output_file`**: path to save JSON data with scores for finetuning
- **`use_fp16`**: Whether to use fp16 for inference. Default: True
- **`devices`**: Devices to use for inference. Default: None, multiple values allowed
- **`trust_remote_code`**: Trust remote code. Default: False
- **`reranker_name_or_path`**: The reranker name or path. Default: None
- **`reranker_model_class`**: The reranker model class. Available classes: ['auto', 'encoder-only-base', 'decoder-only-base', 'decoder-only-layerwise', 'decoder-only-lightweight']. Default: auto
- **`reranker_peft_path`**: The reranker peft path. Default: None
- **`use_bf16`**: Whether to use bf16 for inference. Default: False
- **`query_instruction_for_rerank`**: Instruction for query. Default: None
- **`query_instruction_format_for_rerank`**: Format for query instruction. Default: {{}{}}
- **`passage_instruction_for_rerank`**: Instruction for passage. Default: None
- **`passage_instruction_format_for_rerank`**: Format for passage instruction. Default: {{}{}}
- **`cache_dir`**: Cache directory for models. Default: None
- **`reranker_batch_size`**: Batch size for inference. Default: 3000
- **`reranker_query_max_length`**: Max length for reranking queries. Default: None
- **`reranker_max_length`**: Max length for reranking. Default: 512
- **`normalize`**: Whether to normalize the reranking scores. Default: False
- **`prompt`**: The prompt for the reranker. Default: None
- **`cutoff_layers`**: The output layers of layerwise/lightweight reranker. Default: None
- **`compress_ratio`**: The compress ratio of lightweight reranker. Default: 1
- **`compress_layers`**: The compress layers of lightweight reranker. Default: None, multiple values allowed

### Split Data by Length

You can split the data using the following command:

```shell
python split_data_by_length.py \
--input_path train_data \
--output_dir train_data_split \
--cache_dir .cache \
--log_name .split_log \
--length_list 0 500 1000 2000 3000 4000 5000 6000 7000 \
--model_name_or_path BAAI/bge-m3 \
--num_proc 16 \
--overwrite False
```

- **`input_path`**: The path of input data. (Required)
- **`output_dir`**: The directory of output data. (Required)
- **`cache_dir`**: The cache directory. Default: None
- **`log_name`**: The name of the log file. Default: `.split_log`, which will be saved to `output_dir`
- **`length_list`**: The length list to split. Default: [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
- **`model_name_or_path`**: The model name or path of the tokenizer. Default: `BAAI/bge-m3`
- **`num_proc`**: The number of processes. Default: 16
- **`overwrite`**: Whether to overwrite the output file. Default: False