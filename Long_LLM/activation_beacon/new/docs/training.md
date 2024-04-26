# Training

## Data
You should download the data for fine-tuning & evaluation then untar the file at anywhere you prefer, e.g. `/data`, which results in a folder `/data/activation-beacon-new`:
```bash
# feel free to alternate /data to your prefered location
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/activation-beacon-new.tar.gz?download=true -O /data/activation-beacon-new.tar.gz

cd /data
tar -xzvf activation-beacon-new.tar.gz
```

**IMPORTANT NOTE**

For any path specified for `train_data` and `eval_data`: if it is prefixed with `activation-beacon:`, it will be solved to the relative path against [`data_root`](../src/args.py). 
  - e.g. `activation-beacon:lm/pg19.json` becomes `${data_root}/lm/pg19.json`
  - you can modify the default value of [`data_root`](../src/args.py), so that you don't need to type it for each command.


## Training
Below is the script to train Activation Beacon for Llama-2 with Deepspeed Zero3 and chat template. **The training script for Mistral will be released in future.**

```bash
cd new

torchrun --nproc_per_node 8 -m main.train \
--data_root /data/activation-beacon-new \
--output_dir data/outputs/activation-beacon-llama2-chat-7b \
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
--save_strategy steps \
--max_steps 10000 \
--save_steps 10000 \
--logging_steps 50 \
--chat_template llama-2 \
--group_by_stride strict \
--deepspeed data/deepspeed/stage3.json
```
