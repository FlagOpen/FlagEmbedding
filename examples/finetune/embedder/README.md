# Finetune

In this example, we show how to finetune the embedder with your data.

- [1. Installation](#1-Installation)
- [2. Data format](#2-Data-format)
  - [Hard Negatives](#Hard-Negatives)
  - [Teacher Scores](#Teacher-Scores)
- [3. Train](#3-Train)
  - [(1) standard model](#1-standard-model)
  - [(2) bge-m3](#2-bge-m3)
  - [(3) bge-multilingual-gemma2](#3-bge-multilingual-gemma2)
  - [(4) bge-en-icl](#4-bge-en-icl)

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
{"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[int], "neg_scores": List[int], "prompt": str, "type": str}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts. `pos_scores` is a list of scores corresponding to the `query` and `pos`, `neg_scores` is a list of scores corresponding to the `query` and `neg`, if you don't use knowledge distillation, it can be ignored. `prompt` is the prompt used for the query, it will cover `query_instruction_for_retrieval`. `type` is used for `bge-en-icl`,  it includes `normal`, `symmetric_class`, `symmetric_clustering`, .etc. If you have no negative texts for a query, you can random sample some from the entire corpus as the negatives.

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

Detailed examples of various fine-tuning can be found in the bash files located in the corresponding folders. Here, we simply provide the training methods for the `standard model`, `bge-m3`, `bge-multilingual-gemma2` and `bge-en-icl`.

Here are some import arguments:

- **`model_name_or_path`**: The model checkpoint for initialization.
- **`config_name`**: Pretrained config name or path if not the same as model_name.
- **`tokenizer_name`**: Pretrained tokenizer name or path if not the same as model_name.
- **`cache_dir`**: Where do you want to store the pre-trained models downloaded from s3.
- **`trust_remote_code`**: Trust remote code
- **`token`**: The token to use when accessing the model.
- **`train_data`**: One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the training data. Argument type: multiple.
- **`cache_path`**: Where do you want to store the cached data.
- **`train_group_size`**: (No metadata provided)
- **`query_max_len`**: The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated.
- **`passage_max_len`**: The maximum total input sequence length after tokenization for passage. Sequences longer than this will be truncated.
- **`pad_to_multiple_of`**: If set will pad the sequence to be a multiple of the provided value.
- **`max_example_num_per_dataset`**: The max number of examples for each dataset.
- **`query_instruction_for_retrieval`**: Instruction for query.
- **`query_instruction_format`**: Format for query instruction.
- **`knowledge_distillation`**: Use knowledge distillation when `pos_scores: List[float]` and `neg_scores: List[float]` are in features of training data.
- **`passage_instruction_for_retrieval`**: Instruction for passage.
- **`passage_instruction_format`**: Format for passage instruction.
- **`shuffle_ratio`**: The ratio of shuffling the text.
- **`same_dataset_within_batch`**: All samples in the same batch comes from the same dataset.
- **`small_threshold`**: The threshold of small dataset. All small dataset in the same directory will be merged into one dataset.
- **`drop_threshold`**: The threshold for dropping merged small dataset. If the number of examples in the merged small dataset is less than this threshold, it will be dropped.
- **`negatives_cross_device`**: Share negatives across devices.
- **`temperature`**: Temperature used for similarity score.
- **`fix_position_embedding`**: Freeze the parameters of position embeddings.
- **`sentence_pooling_method`**: The pooling method. Available options: cls, mean, last_token. Default: cls.
- **`normalize_embeddings`**: Whether to normalize the embeddings.
- **`sub_batch_size`**: Sub batch size for training.
- **`kd_loss_type`**: The loss type for knowledge distillation. Available options: kl_div, m3_kd_loss. Default: kl_div.

### (1) standard model

```shell
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path BAAI/bge-large-en-v1.5 \
    --cache_dir ./cache/model \
    --train_data ./example_data/retrieval \
    			 ./example_data/sts/sts.jsonl \
    			 ./example_data/classification-no_in_batch_neg \
    			 ./example_data/clustering-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
	--output_dir ./test_encoder_only_base_bge-large-en-v1.5 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div
```

### (2) bge-m3

```shell
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data ./example_data/retrieval \
    			 ./example_data/sts/sts.jsonl \
    			 ./example_data/classification-no_in_batch_neg \
    			 ./example_data/clustering-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir ./test_encoder_only_m3_bge-m3_sd \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0
```

Here are some new arguments:

- **`colbert_dim`**: Dim of colbert linear
- **`unified_finetuning`**: Use unify fine-tuning
- **`use_self_distill`**: Use self-distill when using unify fine-tuning
- **`fix_encoder`**: Freeze the parameters of encoder
- **`self_distill_start_step`**: Num of step when using self-distill

### (3) bge-multilingual-gemma2

```shell
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
	--model_name_or_path BAAI/bge-multilingual-gemma2 \
    --cache_dir ./cache/model \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --additional_special_tokens '<instruct>' '<query>' \
    --save_merged_lora_model True \
    --train_data ./example_data/retrieval \
    			 ./example_data/sts/sts.jsonl \
    			 ./example_data/classification-no_in_batch_neg \
    			 ./example_data/clustering-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Given a query, retrieve passages that are relevant to the query.' \
    --query_instruction_format '<instruct>{}\n<query>{}' \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
	--output_dir ./test_decoder_only_base_bge-multilingual-gemma2_sd \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage1.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss
```

Here are some new arguments:

- **`peft_model_path`**: The peft model checkpoint for initialization.
- **`use_lora`**: If passed, will use LORA (low-rank parameter-efficient training) to train the model.
- **`lora_rank`**: The rank of lora.
- **`lora_alpha`**: The alpha parameter of lora.
- **`lora_dropout`**: The dropout rate of lora modules.
- **`target_modules`**: The target modules to apply LORA.
- **`use_flash_attn`**: If passed, will use flash attention to train the model.
- **`use_slow_tokenizer`**: If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).
- **`additional_special_tokens`**: Additional special tokens.
- **`save_merged_lora_model`**: If passed, will merge the lora modules and save the entire model.

### (4) bge-en-icl

```shell
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.embedder.decoder_only.icl \
	--model_name_or_path BAAI/bge-en-icl \
    --cache_dir ./cache/model \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --additional_special_tokens '<instruct>' '<query>' '<response>' \
    --save_merged_lora_model True \
    --train_data ./example_data/retrieval \
    			 ./example_data/sts/sts.jsonl \
    			 ./example_data/classification-no_in_batch_neg \
    			 ./example_data/clustering-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 2048 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Given a query, retrieve passages that are relevant to the query.' \
    --query_instruction_format '<instruct>{}\n<query>{}' \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --example_query_max_len 256 \
    --example_passage_max_len 256 \
    --retrieval_use_examples True \
    --icl_suffix_str '\n<response>' \
    --output_dir ./test_decoder_only_base_bge-en-icl_sd \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage1.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div
```

Here are some new arguments:

- **`peft_model_path`**: The peft model checkpoint for initialization.
- **`use_lora`**: If passed, will use LORA (low-rank parameter-efficient training) to train the model.
- **`lora_rank`**: The rank of LORA.
- **`lora_alpha`**: The alpha parameter of LORA.
- **`lora_dropout`**: The dropout rate of LORA modules.
- **`target_modules`**: The target modules to apply LORA.
- **`use_flash_attn`**: If passed, will use flash attention to train the model.
- **`use_slow_tokenizer`**: If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).
- **`from_peft`** (no metadata provided)
- **`modules_to_save`** (no metadata provided)
- **`raw_peft`** (no metadata provided)
- **`additional_special_tokens`**: additional special tokens
- **`save_merged_lora_model`**: If passed, will merge the LORA modules and save the entire model.
- **`example_query_max_len`**: The max length of example query.
- **`example_passage_max_len`**: The max length of example passage.
- **`retrieval_use_examples`**: If passed, will use examples for retrieval.
- **`icl_suffix_str`**: The suffix string for ICL dataset.

