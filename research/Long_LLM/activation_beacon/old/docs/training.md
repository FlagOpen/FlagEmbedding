# Training

## Environment
The main dependencies are:
```
pytorch==2.1.2 transformers==4.36.1 accelerate==0.25.0 datasets==2.14.7 numpy==1.26.2 flash-attn==2.4.2
```
You can install our environment with:
```bash
conda env create -f environment.yaml --name activation-beacon
```

*All of our experiments are performed on one 8xA800 machine with CUDA 12.1.*


## Data
You should download the data for fine-tuning & evaluation then untar the file at anywhere you prefer, e.g. `/data`, which results in a folder `/data/activation-beacon`:
```bash
# feel free to alternate /data to your prefered location
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/activation-beacon.tar.gz?download=true -O /data/activation-beacon.tar.gz

cd /data
tar -xzvf activation-beacon.tar.gz
```

**IMPORTANT NOTE**
- For any path specified for `train_data` and `eval_data`: if it is prefixed with `activation-beacon:`, it will be solved to the relative path against [`data_root`](../src/args.py). 
  - e.g. `activation-beacon:lm/pg19.json` becomes `${data_root}/lm/pg19.json`
  - you can modify the default value of [`data_root`](../src/args.py), so that you don't need to type it for each command.

## Command
```bash
torchrun --nproc_per_node 8 -m main.train \
--output_dir data/outputs/activation-beacon-llama2-chat-7b \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--train_data activation-beacon:pretrain/redpajama-sample.json activation-beacon:finetune/longalpaca.json \
--max_length 8192 \
--min_length 1200 \
--max_train_num_per_data 200000 \
--num_train_epochs 1 \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_stride_mix step-random \
--beacon_attn step-expansion \
--beacon_attend_previous \
--beacon_ratio 2 2 2 2 2 4 4 4 4 4 8 8 16 16 32 32 64 128 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--group_by_length \
--gradient_checkpointing \
--save_strategy steps \
--max_steps 10000 \
--save_steps 10000 \
--logging_steps 50 \
--deepspeed data/deepspeed/stage2.json


# Evaluation
for model in data/outputs/activation-beacon-llama2-chat-7b/*
do
# 100K perplexity
torchrun --nproc_per_node 8 -m main.eval_lm --model_name_or_path $model --max_length 100000 --beacon_ratio 32 --min_length 400000 --enable_beacon --stride 0
# 400K perplexity
torchrun --nproc_per_node 8 -m main.eval_lm --model_name_or_path $model --max_length 400000 --beacon_ratio 128 --min_length 400000 --enable_beacon --stride 0
# LongBench
torchrun --nproc_per_node 8 -m main.eval_longbench --model_name_or_path $model --dataset_names narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum lcc repobench-p --max_length 15500 --enable_beacon
# Topic Retrieval
torchrun --nproc_per_node 8 -m main.eval_longeval --model_name_or_path $model --enable_beacon
done
```