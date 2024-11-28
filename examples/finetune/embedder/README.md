# Finetune

In this example, we show how to finetune the embedder with your data.

- [Finetune](#finetune)
  - [1. Installation](#1-installation)
  - [2. Data format](#2-data-format)
    - [Hard Negatives](#hard-negatives)
    - [Teacher Scores](#teacher-scores)
  - [3. Train](#3-train)
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

```bash
torchrun --nproc_per_node 1 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path BAAI/bge-large-en-v1.5 \
    --train_data ./bge_finetune_data/finetune_data_minedHN.jsonl \
    --temperature 0.02 \
    --output_dir ./FT-1125-bge-large-en-v1.5 \
    --save_steps 250 \
    --per_device_train_batch_size 4 \
    --logging_steps 50 \
    --query_max_len 512 \
    --passage_max_len 64 \
    --train_group_size 8 \
    --cache_dir ./cache/model \
    --cache_path ./cache/data \
    --query_max_len 512 \
    --passage_max_len 64 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --negatives_cross_device
```

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



python -m FlagEmbedding.finetune.embedder.encoder_only.base --help
usage: __main__.py [-h] --model_name_or_path MODEL_NAME_OR_PATH [--config_name CONFIG_NAME] [--tokenizer_name TOKENIZER_NAME]
                   [--cache_dir CACHE_DIR] [--trust_remote_code [TRUST_REMOTE_CODE]] [--token TOKEN]
                   [--train_data TRAIN_DATA [TRAIN_DATA ...]] [--cache_path CACHE_PATH] [--train_group_size TRAIN_GROUP_SIZE]
                   [--query_max_len QUERY_MAX_LEN] [--passage_max_len PASSAGE_MAX_LEN] [--pad_to_multiple_of PAD_TO_MULTIPLE_OF]
                   [--max_example_num_per_dataset MAX_EXAMPLE_NUM_PER_DATASET]
                   [--query_instruction_for_retrieval QUERY_INSTRUCTION_FOR_RETRIEVAL]
                   [--query_instruction_format QUERY_INSTRUCTION_FORMAT] [--knowledge_distillation [KNOWLEDGE_DISTILLATION]]
                   [--passage_instruction_for_retrieval PASSAGE_INSTRUCTION_FOR_RETRIEVAL]
                   [--passage_instruction_format PASSAGE_INSTRUCTION_FORMAT] [--shuffle_ratio SHUFFLE_RATIO]
                   [--same_dataset_within_batch [SAME_DATASET_WITHIN_BATCH]] [--small_threshold SMALL_THRESHOLD]
                   [--drop_threshold DROP_THRESHOLD] --output_dir OUTPUT_DIR [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
                   [--do_train [DO_TRAIN]] [--do_eval [DO_EVAL]] [--do_predict [DO_PREDICT]] [--eval_strategy {no,steps,epoch}]
                   [--prediction_loss_only [PREDICTION_LOSS_ONLY]] [--per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE]
                   [--per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE]
                   [--per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE] [--per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE]
                   [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                   [--eval_accumulation_steps EVAL_ACCUMULATION_STEPS] [--eval_delay EVAL_DELAY]
                   [--torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS] [--learning_rate LEARNING_RATE]
                   [--weight_decay WEIGHT_DECAY] [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                   [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM] [--num_train_epochs NUM_TRAIN_EPOCHS]
                   [--max_steps MAX_STEPS]
                   [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}]
                   [--lr_scheduler_kwargs LR_SCHEDULER_KWARGS] [--warmup_ratio WARMUP_RATIO] [--warmup_steps WARMUP_STEPS]
                   [--log_level {detail,debug,info,warning,error,critical,passive}]
                   [--log_level_replica {detail,debug,info,warning,error,critical,passive}]
                   [--log_on_each_node [LOG_ON_EACH_NODE]] [--no_log_on_each_node] [--logging_dir LOGGING_DIR]
                   [--logging_strategy {no,steps,epoch}] [--logging_first_step [LOGGING_FIRST_STEP]]
                   [--logging_steps LOGGING_STEPS] [--logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]]
                   [--no_logging_nan_inf_filter] [--save_strategy {no,steps,epoch}] [--save_steps SAVE_STEPS]
                   [--save_total_limit SAVE_TOTAL_LIMIT] [--save_safetensors [SAVE_SAFETENSORS]] [--no_save_safetensors]
                   [--save_on_each_node [SAVE_ON_EACH_NODE]] [--save_only_model [SAVE_ONLY_MODEL]]
                   [--restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]] [--no_cuda [NO_CUDA]]
                   [--use_cpu [USE_CPU]] [--use_mps_device [USE_MPS_DEVICE]] [--seed SEED] [--data_seed DATA_SEED]
                   [--jit_mode_eval [JIT_MODE_EVAL]] [--use_ipex [USE_IPEX]] [--bf16 [BF16]] [--fp16 [FP16]]
                   [--fp16_opt_level FP16_OPT_LEVEL] [--half_precision_backend {auto,apex,cpu_amp}]
                   [--bf16_full_eval [BF16_FULL_EVAL]] [--fp16_full_eval [FP16_FULL_EVAL]] [--tf32 TF32]
                   [--local_rank LOCAL_RANK] [--ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl}] [--tpu_num_cores TPU_NUM_CORES]
                   [--tpu_metrics_debug [TPU_METRICS_DEBUG]] [--debug DEBUG [DEBUG ...]]
                   [--dataloader_drop_last [DATALOADER_DROP_LAST]] [--eval_steps EVAL_STEPS]
                   [--dataloader_num_workers DATALOADER_NUM_WORKERS] [--dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR]
                   [--past_index PAST_INDEX] [--run_name RUN_NAME] [--disable_tqdm DISABLE_TQDM]
                   [--remove_unused_columns [REMOVE_UNUSED_COLUMNS]] [--no_remove_unused_columns]
                   [--label_names LABEL_NAMES [LABEL_NAMES ...]] [--load_best_model_at_end [LOAD_BEST_MODEL_AT_END]]
                   [--metric_for_best_model METRIC_FOR_BEST_MODEL] [--greater_is_better GREATER_IS_BETTER]
                   [--ignore_data_skip [IGNORE_DATA_SKIP]] [--fsdp FSDP] [--fsdp_min_num_params FSDP_MIN_NUM_PARAMS]
                   [--fsdp_config FSDP_CONFIG] [--fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP]
                   [--accelerator_config ACCELERATOR_CONFIG] [--deepspeed DEEPSPEED]
                   [--label_smoothing_factor LABEL_SMOOTHING_FACTOR]
                   [--optim {adamw_hf,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo}]
                   [--optim_args OPTIM_ARGS] [--adafactor [ADAFACTOR]] [--group_by_length [GROUP_BY_LENGTH]]
                   [--length_column_name LENGTH_COLUMN_NAME] [--report_to REPORT_TO]
                   [--ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS] [--ddp_bucket_cap_mb DDP_BUCKET_CAP_MB]
                   [--ddp_broadcast_buffers DDP_BROADCAST_BUFFERS] [--dataloader_pin_memory [DATALOADER_PIN_MEMORY]]
                   [--no_dataloader_pin_memory] [--dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS]]
                   [--skip_memory_metrics [SKIP_MEMORY_METRICS]] [--no_skip_memory_metrics]
                   [--use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]] [--push_to_hub [PUSH_TO_HUB]]
                   [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--hub_model_id HUB_MODEL_ID]
                   [--hub_strategy {end,every_save,checkpoint,all_checkpoints}] [--hub_token HUB_TOKEN]
                   [--hub_private_repo [HUB_PRIVATE_REPO]] [--hub_always_push [HUB_ALWAYS_PUSH]]
                   [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                   [--gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS]
                   [--include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]]
                   [--eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES]] [--no_eval_do_concat_batches]
                   [--fp16_backend {auto,apex,cpu_amp}] [--evaluation_strategy {no,steps,epoch}]
                   [--push_to_hub_model_id PUSH_TO_HUB_MODEL_ID] [--push_to_hub_organization PUSH_TO_HUB_ORGANIZATION]
                   [--push_to_hub_token PUSH_TO_HUB_TOKEN] [--mp_parameters MP_PARAMETERS]
                   [--auto_find_batch_size [AUTO_FIND_BATCH_SIZE]] [--full_determinism [FULL_DETERMINISM]]
                   [--torchdynamo TORCHDYNAMO] [--ray_scope RAY_SCOPE] [--ddp_timeout DDP_TIMEOUT]
                   [--torch_compile [TORCH_COMPILE]] [--torch_compile_backend TORCH_COMPILE_BACKEND]
                   [--torch_compile_mode TORCH_COMPILE_MODE] [--dispatch_batches DISPATCH_BATCHES]
                   [--split_batches SPLIT_BATCHES] [--include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND]]
                   [--include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]] [--neftune_noise_alpha NEFTUNE_NOISE_ALPHA]
                   [--optim_target_modules OPTIM_TARGET_MODULES] [--batch_eval_metrics [BATCH_EVAL_METRICS]]
                   [--eval_on_start [EVAL_ON_START]] [--eval_use_gather_object [EVAL_USE_GATHER_OBJECT]]
                   [--negatives_cross_device [NEGATIVES_CROSS_DEVICE]] [--temperature TEMPERATURE]
                   [--fix_position_embedding [FIX_POSITION_EMBEDDING]] [--sentence_pooling_method {cls,mean,last_token}]
                   [--normalize_embeddings [NORMALIZE_EMBEDDINGS]] [--no_normalize_embeddings] [--sub_batch_size SUB_BATCH_SIZE]
                   [--kd_loss_type {kl_div,m3_kd_loss}]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model checkpoint for initialization. (default: None)
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name. (default: None)
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name. (default: None)
  --cache_dir CACHE_DIR
                        Where do you want to store the pre-trained models downloaded from s3. (default: None)
  --trust_remote_code [TRUST_REMOTE_CODE]
                        Trust remote code (default: False)
  --token TOKEN         The token to use when accessing the model. (default: None)
  --train_data TRAIN_DATA [TRAIN_DATA ...]
                        One or more paths to training data. `query: str`, `pos: List[str]`, `neg: List[str]` are required in the
                        training data. (default: None)
  --cache_path CACHE_PATH
                        Where do you want to store the cached data (default: None)
  --train_group_size TRAIN_GROUP_SIZE
  --query_max_len QUERY_MAX_LEN
                        The maximum total input sequence length after tokenization for passage. Sequences longer than this will
                        be truncated. (default: 32)
  --passage_max_len PASSAGE_MAX_LEN
                        The maximum total input sequence length after tokenization for passage. Sequences longer than this will
                        be truncated. (default: 128)
  --pad_to_multiple_of PAD_TO_MULTIPLE_OF
                        If set will pad the sequence to be a multiple of the provided value. (default: None)
  --max_example_num_per_dataset MAX_EXAMPLE_NUM_PER_DATASET
                        the max number of examples for each dataset (default: 100000000)
  --query_instruction_for_retrieval QUERY_INSTRUCTION_FOR_RETRIEVAL
                        instruction for query (default: None)
  --query_instruction_format QUERY_INSTRUCTION_FORMAT
                        format for query instruction (default: {}{})
  --knowledge_distillation [KNOWLEDGE_DISTILLATION]
                        Use knowledge distillation when `pos_scores: List[float]` and `neg_scores: List[float]` are in features
                        of training data (default: False)
  --passage_instruction_for_retrieval PASSAGE_INSTRUCTION_FOR_RETRIEVAL
                        instruction for passage (default: None)
  --passage_instruction_format PASSAGE_INSTRUCTION_FORMAT
                        format for passage instruction (default: {}{})
  --shuffle_ratio SHUFFLE_RATIO
                        The ratio of shuffling the text (default: 0.0)
  --same_dataset_within_batch [SAME_DATASET_WITHIN_BATCH]
                        All samples in the same batch comes from the same dataset. (default: False)
  --small_threshold SMALL_THRESHOLD
                        The threshold of small dataset. All small dataset in the same directory will be merged into one dataset.
                        (default: 0)
  --drop_threshold DROP_THRESHOLD
                        The threshold for dropping merged small dataset. If the number of examples in the merged small dataset
                        is less than this threshold, it will be dropped. (default: 0)
  --output_dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written. (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output directory. Use this to continue training if output_dir points to a
                        checkpoint directory. (default: False)
  --do_train [DO_TRAIN]
                        Whether to run training. (default: False)
  --do_eval [DO_EVAL]   Whether to run eval on the dev set. (default: False)
  --do_predict [DO_PREDICT]
                        Whether to run predictions on the test set. (default: False)
  --eval_strategy {no,steps,epoch}
                        The evaluation strategy to use. (default: no)
  --prediction_loss_only [PREDICTION_LOSS_ONLY]
                        When performing evaluation and predictions, only returns the loss. (default: False)
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size per GPU/TPU/MPS/NPU core/CPU for training. (default: 8)
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation. (default: 8)
  --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
                        Deprecated, the use of `--per_device_train_batch_size` is preferred. Batch size per GPU/TPU core/CPU for
                        training. (default: None)
  --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
                        Deprecated, the use of `--per_device_eval_batch_size` is preferred. Batch size per GPU/TPU core/CPU for
                        evaluation. (default: None)
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass. (default: 1)
  --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
                        Number of predictions steps to accumulate before moving the tensors to the CPU. (default: None)
  --eval_delay EVAL_DELAY
                        Number of epochs or steps to wait for before the first evaluation can be performed, depending on the
                        eval_strategy. (default: 0)
  --torch_empty_cache_steps TORCH_EMPTY_CACHE_STEPS
                        Number of steps to wait before calling `torch.<device>.empty_cache()`.This can help avoid CUDA out-of-
                        memory errors by lowering peak VRAM usage at a cost of about [10{'option_strings': ['--
                        torch_empty_cache_steps'], 'dest': 'torch_empty_cache_steps', 'nargs': None, 'const': None, 'default':
                        None, 'type': 'int', 'choices': None, 'required': False, 'help': 'Number of steps to wait before calling
                        `torch.<device>.empty_cache()`.This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage
                        at a cost of about [10% slower performance](https://github.com/huggingface/transformers/issues/31372).If
                        left unset or set to None, cache will not be emptied.', 'metavar': None, 'container':
                        <argparse._ArgumentGroup object at 0x7f5d265be530>, 'prog': '__main__.py'}lower
                        performance](https://github.com/huggingface/transformers/issues/31372).If left unset or set to None,
                        cache will not be emptied. (default: None)
  --learning_rate LEARNING_RATE
                        The initial learning rate for AdamW. (default: 5e-05)
  --weight_decay WEIGHT_DECAY
                        Weight decay for AdamW if we apply some. (default: 0.0)
  --adam_beta1 ADAM_BETA1
                        Beta1 for AdamW optimizer (default: 0.9)
  --adam_beta2 ADAM_BETA2
                        Beta2 for AdamW optimizer (default: 0.999)
  --adam_epsilon ADAM_EPSILON
                        Epsilon for AdamW optimizer. (default: 1e-08)
  --max_grad_norm MAX_GRAD_NORM
                        Max gradient norm. (default: 1.0)
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Total number of training epochs to perform. (default: 3.0)
  --max_steps MAX_STEPS
                        If > 0: set total number of training steps to perform. Override num_train_epochs. (default: -1)
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau,cosine_with_min_lr,warmup_stable_decay}
                        The scheduler type to use. (default: linear)
  --lr_scheduler_kwargs LR_SCHEDULER_KWARGS
                        Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts.
                        (default: {})
  --warmup_ratio WARMUP_RATIO
                        Linear warmup over warmup_ratio fraction of total steps. (default: 0.0)
  --warmup_steps WARMUP_STEPS
                        Linear warmup over warmup_steps. (default: 0)
  --log_level {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',
                        'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets
                        the application set the level. Defaults to 'passive'. (default: passive)
  --log_level_replica {detail,debug,info,warning,error,critical,passive}
                        Logger log level to use on replica nodes. Same choices and defaults as ``log_level`` (default: warning)
  --log_on_each_node [LOG_ON_EACH_NODE]
                        When doing a multinode distributed training, whether to log once per node or just once on the main node.
                        (default: True)
  --no_log_on_each_node
                        When doing a multinode distributed training, whether to log once per node or just once on the main node.
                        (default: False)
  --logging_dir LOGGING_DIR
                        Tensorboard log dir. (default: None)
  --logging_strategy {no,steps,epoch}
                        The logging strategy to use. (default: steps)
  --logging_first_step [LOGGING_FIRST_STEP]
                        Log the first global_step (default: False)
  --logging_steps LOGGING_STEPS
                        Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, will be
                        interpreted as ratio of total training steps. (default: 500)
  --logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]
                        Filter nan and inf losses for logging. (default: True)
  --no_logging_nan_inf_filter
                        Filter nan and inf losses for logging. (default: False)
  --save_strategy {no,steps,epoch}
                        The checkpoint save strategy to use. (default: steps)
  --save_steps SAVE_STEPS
                        Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than
                        1, will be interpreted as ratio of total training steps. (default: 500)
  --save_total_limit SAVE_TOTAL_LIMIT
                        If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
                        `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to
                        `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for
                        `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be
                        retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`, it is
                        possible that two checkpoints are saved: the last one and the best one (if they are different). Default
                        is unlimited checkpoints (default: None)
  --save_safetensors [SAVE_SAFETENSORS]
                        Use safetensors saving and loading for state dicts instead of default torch.load and torch.save.
                        (default: True)
  --no_save_safetensors
                        Use safetensors saving and loading for state dicts instead of default torch.load and torch.save.
                        (default: False)
  --save_on_each_node [SAVE_ON_EACH_NODE]
                        When doing multi-node distributed training, whether to save models and checkpoints on each node, or only
                        on the main one (default: False)
  --save_only_model [SAVE_ONLY_MODEL]
                        When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.Note
                        that when this is true, you won't be able to resume training from checkpoint.This enables you to save
                        storage by not storing the optimizer, scheduler & rng state.You can only load the model using
                        from_pretrained with this option set to True. (default: False)
  --restore_callback_states_from_checkpoint [RESTORE_CALLBACK_STATES_FROM_CHECKPOINT]
                        Whether to restore the callback states from the checkpoint. If `True`, will override callbacks passed to
                        the `Trainer` if they exist in the checkpoint. (default: False)
  --no_cuda [NO_CUDA]   This argument is deprecated. It will be removed in version 5.0 of 🤗 Transformers. (default: False)
  --use_cpu [USE_CPU]   Whether or not to use cpu. If set to False, we will use cuda/tpu/mps/npu device if available. (default:
                        False)
  --use_mps_device [USE_MPS_DEVICE]
                        This argument is deprecated. `mps` device will be used if available similar to `cuda` device. It will be
                        removed in version 5.0 of 🤗 Transformers (default: False)
  --seed SEED           Random seed that will be set at the beginning of training. (default: 42)
  --data_seed DATA_SEED
                        Random seed to be used with data samplers. (default: None)
  --jit_mode_eval [JIT_MODE_EVAL]
                        Whether or not to use PyTorch jit trace for inference (default: False)
  --use_ipex [USE_IPEX]
                        Use Intel extension for PyTorch when it is available, installation: 'https://github.com/intel/intel-
                        extension-for-pytorch' (default: False)
  --bf16 [BF16]         Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture
                        or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change. (default: False)
  --fp16 [FP16]         Whether to use fp16 (mixed) precision instead of 32-bit (default: False)
  --fp16_opt_level FP16_OPT_LEVEL
                        For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at
                        https://nvidia.github.io/apex/amp.html (default: O1)
  --half_precision_backend {auto,apex,cpu_amp}
                        The backend to be used for half precision. (default: auto)
  --bf16_full_eval [BF16_FULL_EVAL]
                        Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may
                        change. (default: False)
  --fp16_full_eval [FP16_FULL_EVAL]
                        Whether to use full float16 evaluation instead of 32-bit (default: False)
  --tf32 TF32           Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental
                        API and it may change. (default: None)
  --local_rank LOCAL_RANK
                        For distributed training: local_rank (default: -1)
  --ddp_backend {nccl,gloo,mpi,ccl,hccl,cncl}
                        The backend to be used for distributed training (default: None)
  --tpu_num_cores TPU_NUM_CORES
                        TPU: Number of TPU cores (automatically passed by launcher script) (default: None)
  --tpu_metrics_debug [TPU_METRICS_DEBUG]
                        Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics
                        (default: False)
  --debug DEBUG [DEBUG ...]
                        Whether or not to enable debug mode. Current options: `underflow_overflow` (Detect underflow and
                        overflow in activations and weights), `tpu_metrics_debug` (print debug metrics on TPU). (default: None)
  --dataloader_drop_last [DATALOADER_DROP_LAST]
                        Drop the last incomplete batch if it is not divisible by the batch size. (default: False)
  --eval_steps EVAL_STEPS
                        Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. If smaller than 1,
                        will be interpreted as ratio of total training steps. (default: None)
  --dataloader_num_workers DATALOADER_NUM_WORKERS
                        Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in
                        the main process. (default: 0)
  --dataloader_prefetch_factor DATALOADER_PREFETCH_FACTOR
                        Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers
                        batches prefetched across all workers. Default is 2 for PyTorch < 2.0.0 and otherwise None. (default:
                        None)
  --past_index PAST_INDEX
                        If >=0, uses the corresponding part of the output as the past state for next step. (default: -1)
  --run_name RUN_NAME   An optional descriptor for the run. Notably used for wandb, mlflow and comet logging. (default: None)
  --disable_tqdm DISABLE_TQDM
                        Whether or not to disable the tqdm progress bars. (default: None)
  --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
                        Remove columns not required by the model when using an nlp.Dataset. (default: True)
  --no_remove_unused_columns
                        Remove columns not required by the model when using an nlp.Dataset. (default: False)
  --label_names LABEL_NAMES [LABEL_NAMES ...]
                        The list of keys in your dictionary of inputs that correspond to the labels. (default: None)
  --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
                        Whether or not to load the best model found during training at the end of training. When this option is
                        enabled, the best checkpoint will always be saved. See `save_total_limit` for more. (default: False)
  --metric_for_best_model METRIC_FOR_BEST_MODEL
                        The metric to use to compare two different models. (default: None)
  --greater_is_better GREATER_IS_BETTER
                        Whether the `metric_for_best_model` should be maximized or not. (default: None)
  --ignore_data_skip [IGNORE_DATA_SKIP]
                        When resuming training, whether or not to skip the first epochs and batches to get to the same training
                        data. (default: False)
  --fsdp FSDP           Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training
                        only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add CPU-offload
                        to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op offload`. You can
                        add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard auto_wrap` or
                        `shard_grad_op auto_wrap`. (default: )
  --fsdp_min_num_params FSDP_MIN_NUM_PARAMS
                        This parameter is deprecated. FSDP's minimum number of parameters for Default Auto Wrapping. (useful
                        only when `fsdp` field is passed). (default: 0)
  --fsdp_config FSDP_CONFIG
                        Config to be used with FSDP (Pytorch Fully Sharded Data Parallel). The value is either a fsdp json
                        config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`. (default: None)
  --fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
                        This parameter is deprecated. Transformer layer class name (case-sensitive) to wrap, e.g, `BertLayer`,
                        `GPTJBlock`, `T5Block` .... (useful only when `fsdp` flag is passed). (default: None)
  --accelerator_config ACCELERATOR_CONFIG
                        Config to be used with the internal Accelerator object initializtion. The value is either a accelerator
                        json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`. (default:
                        None)
  --deepspeed DEEPSPEED
                        Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already
                        loaded json file as a dict (default: None)
  --label_smoothing_factor LABEL_SMOOTHING_FACTOR
                        The label smoothing epsilon to apply (zero means no label smoothing). (default: 0.0)
  --optim {adamw_hf,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo}
                        The optimizer to use. (default: adamw_torch)
  --optim_args OPTIM_ARGS
                        Optional arguments to supply to optimizer. (default: None)
  --adafactor [ADAFACTOR]
                        Whether or not to replace AdamW by Adafactor. (default: False)
  --group_by_length [GROUP_BY_LENGTH]
                        Whether or not to group samples of roughly the same length together when batching. (default: False)
  --length_column_name LENGTH_COLUMN_NAME
                        Column name with precomputed lengths to use when grouping by length. (default: length)
  --report_to REPORT_TO
                        The list of integrations to report the results and logs to. (default: None)
  --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
                        When using distributed training, the value of the flag `find_unused_parameters` passed to
                        `DistributedDataParallel`. (default: None)
  --ddp_bucket_cap_mb DDP_BUCKET_CAP_MB
                        When using distributed training, the value of the flag `bucket_cap_mb` passed to
                        `DistributedDataParallel`. (default: None)
  --ddp_broadcast_buffers DDP_BROADCAST_BUFFERS
                        When using distributed training, the value of the flag `broadcast_buffers` passed to
                        `DistributedDataParallel`. (default: None)
  --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
                        Whether or not to pin memory for DataLoader. (default: True)
  --no_dataloader_pin_memory
                        Whether or not to pin memory for DataLoader. (default: False)
  --dataloader_persistent_workers [DATALOADER_PERSISTENT_WORKERS]
                        If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
                        This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will
                        increase RAM usage. (default: False)
  --skip_memory_metrics [SKIP_MEMORY_METRICS]
                        Whether or not to skip adding of memory profiler reports to metrics. (default: True)
  --no_skip_memory_metrics
                        Whether or not to skip adding of memory profiler reports to metrics. (default: False)
  --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]
                        Whether or not to use the legacy prediction_loop in the Trainer. (default: False)
  --push_to_hub [PUSH_TO_HUB]
                        Whether or not to upload the trained model to the model hub after training. (default: False)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        The path to a folder with a valid checkpoint for your model. (default: None)
  --hub_model_id HUB_MODEL_ID
                        The name of the repository to keep in sync with the local `output_dir`. (default: None)
  --hub_strategy {end,every_save,checkpoint,all_checkpoints}
                        The hub strategy to use when `--push_to_hub` is activated. (default: every_save)
  --hub_token HUB_TOKEN
                        The token to use to push to the Model Hub. (default: None)
  --hub_private_repo [HUB_PRIVATE_REPO]
                        Whether the model repository is private or not. (default: False)
  --hub_always_push [HUB_ALWAYS_PUSH]
                        Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet. (default: False)
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        If True, use gradient checkpointing to save memory at the expense of slower backward pass. (default:
                        False)
  --gradient_checkpointing_kwargs GRADIENT_CHECKPOINTING_KWARGS
                        Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to
                        `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`. (default: None)
  --include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]
                        Whether or not the inputs will be passed to the `compute_metrics` function. (default: False)
  --eval_do_concat_batches [EVAL_DO_CONCAT_BATCHES]
                        Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`, will instead
                        store them as lists, with each batch kept separate. (default: True)
  --no_eval_do_concat_batches
                        Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`, will instead
                        store them as lists, with each batch kept separate. (default: False)
  --fp16_backend {auto,apex,cpu_amp}
                        Deprecated. Use half_precision_backend instead (default: auto)
  --evaluation_strategy {no,steps,epoch}
                        Deprecated. Use `eval_strategy` instead (default: None)
  --push_to_hub_model_id PUSH_TO_HUB_MODEL_ID
                        The name of the repository to which push the `Trainer`. (default: None)
  --push_to_hub_organization PUSH_TO_HUB_ORGANIZATION
                        The name of the organization in with to which push the `Trainer`. (default: None)
  --push_to_hub_token PUSH_TO_HUB_TOKEN
                        The token to use to push to the Model Hub. (default: None)
  --mp_parameters MP_PARAMETERS
                        Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer (default: )
  --auto_find_batch_size [AUTO_FIND_BATCH_SIZE]
                        Whether to automatically decrease the batch size in half and rerun the training loop again each time a
                        CUDA Out-of-Memory was reached (default: False)
  --full_determinism [FULL_DETERMINISM]
                        Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed training.
                        Important: this will negatively impact the performance, so only use it for debugging. (default: False)
  --torchdynamo TORCHDYNAMO
                        This argument is deprecated, use `--torch_compile_backend` instead. (default: None)
  --ray_scope RAY_SCOPE
                        The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray will
                        then use the last checkpoint of all trials, compare those, and select the best one. However, other
                        options are also available. See the Ray documentation
                        (https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial)
                        for more options. (default: last)
  --ddp_timeout DDP_TIMEOUT
                        Overrides the default timeout for distributed training (value should be given in seconds). (default:
                        1800)
  --torch_compile [TORCH_COMPILE]
                        If set to `True`, the model will be wrapped in `torch.compile`. (default: False)
  --torch_compile_backend TORCH_COMPILE_BACKEND
                        Which backend to use with `torch.compile`, passing one will trigger a model compilation. (default: None)
  --torch_compile_mode TORCH_COMPILE_MODE
                        Which mode to use with `torch.compile`, passing one will trigger a model compilation. (default: None)
  --dispatch_batches DISPATCH_BATCHES
                        Deprecated. Pass {'dispatch_batches':VALUE} to `accelerator_config`. (default: None)
  --split_batches SPLIT_BATCHES
                        Deprecated. Pass {'split_batches':True} to `accelerator_config`. (default: None)
  --include_tokens_per_second [INCLUDE_TOKENS_PER_SECOND]
                        If set to `True`, the speed metrics will include `tgs` (tokens per second per device). (default: False)
  --include_num_input_tokens_seen [INCLUDE_NUM_INPUT_TOKENS_SEEN]
                        If set to `True`, will track the number of input tokens seen throughout training. (May be slower in
                        distributed training) (default: False)
  --neftune_noise_alpha NEFTUNE_NOISE_ALPHA
                        Activates neftune noise embeddings into the model. NEFTune has been proven to drastically improve model
                        performances for instrcution fine-tuning. Check out the original paper here:
                        https://arxiv.org/abs/2310.05914 and the original code here: https://github.com/neelsjain/NEFTune. Only
                        supported for `PreTrainedModel` and `PeftModel` classes. (default: None)
  --optim_target_modules OPTIM_TARGET_MODULES
                        Target modules for the optimizer defined in the `optim` argument. Only used for the GaLore optimizer at
                        the moment. (default: None)
  --batch_eval_metrics [BATCH_EVAL_METRICS]
                        Break eval metrics calculation into batches to save memory. (default: False)
  --eval_on_start [EVAL_ON_START]
                        Whether to run through the entire `evaluation` step at the very beginning of training as a sanity check.
                        (default: False)
  --eval_use_gather_object [EVAL_USE_GATHER_OBJECT]
                        Whether to run recursively gather object in a nested list/tuple/dictionary of objects from all devices.
                        (default: False)
  --negatives_cross_device [NEGATIVES_CROSS_DEVICE]
                        share negatives across devices (default: False)
  --temperature TEMPERATURE
                        temperature used for similarity score (default: 0.02)
  --fix_position_embedding [FIX_POSITION_EMBEDDING]
                        Freeze the parameters of position embeddings (default: False)
  --sentence_pooling_method {cls,mean,last_token}
                        the pooling method. Available options: cls, mean, last_token. Default: cls (default: cls)
  --normalize_embeddings [NORMALIZE_EMBEDDINGS]
                        whether to normalize the embeddings (default: True)
  --no_normalize_embeddings
                        whether to normalize the embeddings (default: False)
  --sub_batch_size SUB_BATCH_SIZE
                        sub batch size for training (default: None)
  --kd_loss_type {kl_div,m3_kd_loss}
                        the loss type for knowledge distillation. Available options: kl_div, m3_kd_loss. Default: kl_div.
                        (default: kl_div)