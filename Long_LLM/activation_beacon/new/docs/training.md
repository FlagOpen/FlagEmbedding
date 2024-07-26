# Training

There are two stages in training:
- Pretrain
  - 1B token from [redpajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) with auto-regressive language modeling
  - Add eos to each document and no packing
  - 20K context length at maximum

- Finetune
  - 5K samples from [LongAlpaca](https://huggingface.co/datasets/Yukang/LongAlpaca-12k), 2K samples from [Booksum](https://huggingface.co/datasets/kmfoda/booksum), 16K synthetic long-context QA data from GPT-3.5, and 5K samples from pretraining data
  - 20K context length at maximum


## Prerequisite

Make sure you have created the environment and downloaded the data according to [README](../README.md).

### Mistral
#### Pretrain
```bash
output_name=beacon-mistral-pretrain

torchrun --nproc_per_node 8 $DDP -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--train_data long-llm:redpajama/train.json \
--min_length 2400 \
--max_length 20000 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 2 4 8 16 32 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--attn_impl flash_attention_2 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy epoch \
--evaluation_strategy steps \
--num_train_epochs 1 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
```

#### Finetune
```bash
output_name=beacon-mistral-finetune

torchrun --nproc_per_node 8 $DDP -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/beacon-mistral-pretrain/* \
--train_data long-llm:gpt/one_detail_book.train.16K.json long-llm:gpt/one_detail_paper.train.16K.json long-llm:longalpaca/train.json long-llm:booksum/train.16K.json long-llm:needle/train.16K.json long-llm:redpajama/train.json[5000] \
--max_length 20000 \
--min_length 7200 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 2 4 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--attn_impl flash_attention_2 \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json \
--chat_template mistral
```

### Llama-3
NOTE: according to our experiment, Llama-3 requires attention sink.

#### Pretrain
```bash
output_name=beacon-llama3-pretrain

torchrun --nproc_per_node 8 $DDP -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
--train_data long-llm:redpajama/train.json \
--min_length 2400 \
--max_length 20000 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 2 4 8 16 32 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--attn_impl flash_attention_2 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy epoch \
--evaluation_strategy steps \
--num_train_epochs 1 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
```

#### Finetune
```bash
output_name=beacon-llama3-finetune

torchrun --nproc_per_node 8 $DDP -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/beacon-llama3-pretrain/* \
--train_data long-llm:gpt/one_detail_book.train.16K.json long-llm:gpt/one_detail_paper.train.16K.json long-llm:longalpaca/train.json long-llm:booksum/train.16K.json long-llm:needle/train.16K.json long-llm:redpajama/train.json[5000] \
--max_length 20000 \
--min_length 7200 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 2 4 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--attn_impl flash_attention_2 \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json \
--chat_template llama-3
```

### Qwen-2
#### Pretrain
```bash
output_name=beacon-qwen2-pretrain

torchrun --nproc_per_node 8 $DDP -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path Qwen/Qwen2-7B-Instruct \
--train_data long-llm:redpajama/train.json \
--min_length 2400 \
--max_length 20000 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 2 4 8 16 32 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--attn_impl flash_attention_2 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy epoch \
--evaluation_strategy steps \
--num_train_epochs 1 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json

```


#### Finetune
```bash
torchrun --nproc_per_node 8 $DDP -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/beacon-qwen2-pretrain/* \
--train_data long-llm:gpt/one_detail_book.train.16K.json long-llm:gpt/one_detail_paper.train.16K.json long-llm:longalpaca/train.json long-llm:booksum/train.16K.json long-llm:needle/train.16K.json long-llm:redpajama/train.json[5000] \
--max_length 20000 \
--min_length 7200 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 2 4 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--attn_impl flash_attention_2 \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json \
--chat_template qwen
```
