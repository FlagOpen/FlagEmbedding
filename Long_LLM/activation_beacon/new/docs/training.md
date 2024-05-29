# Training

There are two training recipes.
- `v1` uses redpajama and longalpaca to train, which is fast and the trained model achieves reasonable performance (but often fails to remember accurate information like needle and passkey).
- `v2` uses slimpajama for pre-training and a mixture of longalpaca, booksum, and synthetic data for fine-tuning, which takes longer time but the trained model achieves substantially better performance.


## Prerequisite

Make sure you have created the environment and downloaded the data according to [README](../README.md).

## V1
```bash
torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/beacon-llama2-chat-7b \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--train_data activation-beacon:redpajama/train.json[200000] activation-beacon:longalpaca/train.json \
--max_length 8192 \
--min_length 1200 \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn step-expansion \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 2 2 2 2 2 4 4 4 4 4 8 8 16 16 32 32 64 128 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--gradient_checkpointing \
--use_reentrant False \
--save_strategy steps \
--max_steps 10000 \
--save_steps 10000 \
--logging_steps 50 \
--chat_template llama-2 \
--group_by_stride strict \
--deepspeed data/deepspeed/stage2.json
```


## V2
### Llama-2
#### Pre-Training
```bash
# prepare 2B data (packing texts from the same source to form sequences of 8K length)
# you only need to run this command once
python -m main.pretrain_data --output_dir data/pretrain/llama-8K_2B --num_token 8192:2b --model_name_or_path meta-llama/Llama-2-7b-chat-hf

output_name=beacon-llama2-7b-chat-pt

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--train_data data/pretrain/llama-8K_2B \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn step-expansion \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 2 4 8 16 32 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy steps \
--evaluation_strategy steps \
--num_train_epochs 1 \
--save_steps 0.49 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
```

#### Fine-Tuning
```bash
# prepare 100M data that are evenly distributed across all domains to prevent forgetting during fine-tuning
python -m main.pretrain_data --output_dir data/pretrain/llama-8K_100M-even --num_token 8192:100m --config data/config/even.json

output_name=beacon-llama2-7b-chat-ft

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/beacon-llama2-7b-chat-pt/checkpoint-xxxxx \
--train_data activation-beacon:gpt/one_detail_book.train.8K.json activation-beacon:gpt/one_detail_paper.train.8K.json activation-beacon:longalpaca/train.json activation-beacon:booksum/train.8K.json activation-beacon:needle/train.8K.json  data/pretrain/llama-8K_100M-even[5000] \
--max_length 10240 \
--min_length 7200 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn step-expansion \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 2 4 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--chat_template llama-2 \
--deepspeed data/deepspeed/stage2.json
```

### Mistral
#### Pre-Training
```bash
# prepare 2B data (packing texts from the same source to form sequences of 16K length)
# you only need to run this command once
python -m main.pretrain_data --output_dir data/pretrain/mistral-16K_2B/ --num_token 16384:2b --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2

output_name=beacon-llama2-7b-chat-pt

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
--train_data data/pretrain/mistral-2B-16K/ \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn step-expansion \
--beacon_attend_prev False \
--beacon_sink_size 1 \
--beacon_ratio 2 4 8 16 32 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy steps \
--save_steps 0.49 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
```

#### Fine-Tuning
```bash
# prepare 100M data that are evenly distributed across all domains to prevent forgetting during fine-tuning
python -m main.pretrain_data --output_dir data/pretrain/mistral-16K_100M-even --num_token 16384:100m --config data/config/even.json --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2

output_name=beacon-mistral-7b-inst-ft

torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/$output_name \
--model_name_or_path data/outputs/beacon-mistral-7b-inst-pt/checkpoint-xxxxx \
--train_data activation-beacon:gpt/one_detail_book.train.16K.json activation-beacon:gpt/one_detail_paper.train.16K.json activation-beacon:longalpaca/train.json activation-beacon:booksum/train.16K.json activation-beacon:needle/train.16K.json data/pretrain/mistral-16K_100M-even[5000] \
--max_length 20480 \
--min_length 7200 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn step-expansion \
--beacon_attend_prev False \
--beacon_sink_size 1 \
--beacon_ratio 2 4 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--learning_rate 1e-5 \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy epoch \
--logging_steps 50 \
--bf16 \
--chat_template mistral \
--deepspeed data/deepspeed/stage2.json
```
