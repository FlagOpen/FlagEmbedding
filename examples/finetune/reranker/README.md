# Finetune

In this example, we show how to finetune the reranker with your data.

- [1. Installation](#1-Installation)
- [2. Data format](#2-Data-format)
  - [Hard Negatives](#Hard-Negatives)
  - [Teacher Scores](#Teacher-Scores)
- [3. Train](#3-Train)
  - [(1) standard model](#1-standard-model)
  - [(2) bge-reranker-v2-gemma](#2-bge-reranker-v2-gemma)
  - [(3) bge-reranker-v2-layerwise-minicpm](#3-bge-reranker-v2-layerwise-minicpm)

## 1. Installation

- **with pip**

```shell
pip install -U FlagEmbedding[finetune]
```

- **from source**

```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .[finetune]
```

For development, install as editable:

```shell
pip install -e .[finetune]
```

## 2. Data format

Train data should be a json file, where each line is a dict like this:

```shell
{"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[int], "neg_scores": List[int], "prompt": str}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts. `pos_scores` is a list of scores corresponding to the `query` and `pos`, `neg_scores` is a list of scores corresponding to the `query` and `neg`, if you don't use knowledge distillation, it can be ignored. `prompt` is the prompt used for the input, input has the following format: `query [sep] passage [sep] prompt`. If you have no negative texts for a query, you can random sample some from the entire corpus as the negatives.

See [example_data](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder/example_data) for more detailed files.

### Hard Negatives

Hard negatives is a widely used method to improve the quality of sentence embedding. You can mine hard negatives following this command:

```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding/scripts
```

```shell
python hn_mine.py \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 
```

- **`input_file`**: json data for finetuning. This script will retrieve top-k documents for each query, and random sample negatives from the top-k documents (not including the positive documents).
- **`output_file`**: path to save JSON data with mined hard negatives for finetuning
- **`negative_number`**: the number of sampled negatives
- **`range_for_sampling`**: where to sample negative. For example, `2-100` means sampling `negative_number` negatives from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top60-300 passages)**
- **`candidate_pool`**: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. The format of this file is the same as [pretrain data](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain#2-data-format). If input a candidate_pool, this script will retrieve negatives from this file.
- **`use_gpu_for_searching`**: whether to use faiss-gpu to retrieve negatives.

### Teacher Scores

Teacher scores can be used for model distillation. You can obtain the scores using the following command:

```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding/scripts
```

```shell
python add_reranker_score.py \
--input_file toy_finetune_data_minedHN.jsonl \
--output_file toy_finetune_data_score.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--devices cuda:0 cuda:1 \
--cache_dir ./cache/model \
--reranker_query_max_length 512 \
--reranker_max_length 1024
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

## 3. Train

Detailed examples of various fine-tuning can be found in the bash files located in the corresponding folders. Here, we simply provide the training methods for the `standard model`, `bge-reranker-v2-gemma` and `bge-reranker-v2-layerwise-minicpm`.

Here are some import arguments:

- **`model_name_or_path`**: The model checkpoint for initialization.
- **`config_name`**: Pretrained config name or path if not the same as model_name. Default: None
- **`tokenizer_name`**: Pretrained tokenizer name or path if not the same as model_name. Default: None
- **`cache_dir`**: Where do you want to store the pre-trained models downloaded from s3. Default: None
- **`trust_remote_code`**: Trust remote code. Default: False
- **`model_type`**: Type of finetune, ['encoder', 'decoder']. Default: 'encoder'
- **`token`**: The token to use when accessing the model. Default: Value from environment variable HF_TOKEN or None if not set
- **`train_data`**: One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the training data. Default: None
- **`cache_path`**: Where do you want to store the cached data. Default: None
- **`train_group_size`**: Default: 8
- **`query_max_len`**: The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated. Default: 32
- **`passage_max_len`**: The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated. Default: 128
- **`max_len`**: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated. Default: 512
- **`pad_to_multiple_of`**: If set, will pad the sequence to be a multiple of the provided value. Default: None
- **`max_example_num_per_dataset`**: The max number of examples for each dataset. Default: 100000000
- **`query_instruction_for_rerank`**: Instruction for query. Default: None
- **`query_instruction_format`**: Format for query instruction. Default: "{}{}"
- **`knowledge_distillation`**: Use knowledge distillation when `pos_scores: List[float]` and `neg_scores: List[float]` are in features of training data. Default: False
- **`passage_instruction_for_rerank`**: Instruction for passage. Default: None
- **`passage_instruction_format`**: Format for passage instruction. Default: "{}{}"
- **`shuffle_ratio`**: The ratio of shuffling the text. Default: 0.0
- **`sep_token`**: The separator token for LLM reranker to discriminate between query and passage. Default: '\n'

### (1) standard model

```shell
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.reranker.encoder_only.base \
	--model_name_or_path BAAI/bge-reranker-v2-m3 \
    --cache_dir ./cache/model \
    --train_data ./example_data/normal/examples.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
	--output_dir ./test_encoder_only_base_bge-reranker-base \
    --overwrite_output_dir \
    --learning_rate 6e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000
```

### (2) bge-reranker-v2-gemma

```shell
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.reranker.decoder_only.base \
	--model_name_or_path BAAI/bge-reranker-v2-gemma \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_flash_attn True \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_merged_lora_model True \
    --model_type decoder \
    --cache_dir ./cache/model \
    --train_data ./example_data/prompt_based/examples.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
    --output_dir ./test_decoder_only_base_bge-reranker-v2-minicpm-layerwise \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000
```

Here are some new arguments:

- **`use_lora`**: If passed, will use LORA (low-rank parameter-efficient training) to train the model.
- **`lora_rank`**: The rank of lora.
- **`lora_alpha`**: The alpha parameter of lora.
- **`lora_dropout`**: The dropout rate of lora modules.
- **`target_modules`**: The target modules to apply LORA.
- **`modules_to_save`**: List of modules that should be saved in the final checkpoint.
- **`use_flash_attn`**: If passed, will use flash attention to train the model.
- **`from_peft`**: (metadata not provided)
- **`raw_peft`**: (metadata not provided)
- **`save_merged_lora_model`**: If passed, will merge the lora modules and save the entire model.

### (3) bge-reranker-v2-layerwise-minicpm

```shell
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.reranker.decoder_only.layerwise \
    --model_name_or_path BAAI/bge-reranker-v2-minicpm-layerwise \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_flash_attn True \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_merged_lora_model True \
    --model_type decoder \
    --model_type from_finetuned_model \
    --start_layer 8 \
    --head_multi True \
    --head_type simple \
    --trust_remote_code True \
    --cache_dir ./cache/model \
    --train_data ./example_data/prompt_based/examples.jsonl \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
	--output_dir ./test_decoder_only_base_bge-reranker-v2-minicpm-layerwise \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000
```

Here are some new arguments:

- **`use_lora`**: If passed, will use LORA (low-rank parameter-efficient training) to train the model.
- **`lora_rank`**: The rank of lora.
- **`lora_alpha`**: The alpha parameter of lora.
- **`lora_dropout`**: The dropout rate of lora modules.
- **`target_modules`**: The target modules to apply LORA.
- **`modules_to_save`**: List of modules that should be saved in the final checkpoint.
- **`use_flash_attn`**: If passed, will use flash attention to train the model.
- **`save_merged_lora_model`**: If passed, will merge the lora modules and save the entire model.
- **`model_type`**: Model type context, which should be one of ['from_raw_model', 'from_finetuned_model'].
- **`start_layer`**: Specifies which layer to start to compute score.
- **`head_multi`**: Indicates whether to use one or multiple classifiers.
- **`head_type`**: The type of the classifier.
