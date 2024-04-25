# Evaluation

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


## Evaluating Beacon Models
```bash
conda activate beacon

data_root=/data/activation-beacon-new
model=namespace-Pt/activation-beacon-mistral-7b

# language modeling perplexity
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 100000 --stride 32768 --model_name_or_path $model

# passkey retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_passkey --data_root $data_root --model_name_or_path $model --chat_template mistral

# needle-in-a-haystack accuracy (remember to set OPENAI_API_KEY in your environmental variable)
torchrun --nproc_per_node 8 -m main.eval_needle --data_root $data_root --model_name_or_path $model --chat_template mistral --gpt_eval

# topic retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_topic --data_root $data_root --model_name_or_path $model --chat_template mistral

# longbench
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 31500 --model_name_or_path $model --chat_template mistral

# infinitebench
torchrun --nproc_per_node 8 -m main.eval_infbench --data_root $data_root --max_length 128000 --model_name_or_path $model --chat_template mistral
```

All evaluation results will be saved at `data/results`.



## Evaluating Full-Attention Models

Full-attention models cannot run with more than 32K context length on a single A800 GPU. Parallel strategies are required. We use [`tensor_parallel`](https://github.com/BlackSamorez/tensor_parallel). You should create anothr environtment while downgrade to `transformers==4.35.1` and install `tensor_parallel`:
```bash
conda create full --clone beacon
pip install transformers==4.35.1 tensor_parallel
```

Then, run the following commands: (feel free to switch `mistralai/Mistral-7B-Instruct-v0.2` to any models on huggingface)

```bash
conda activate full

data_root=/data/activation-beacon-new
model=mistralai/Mistral-7B-Instruct-v0.2

# language modeling perplexity
python -m main.eval_lm  --data_root $data_root --max_length 100000 --stride 32768 --model_name_or_path $model --attn_impl flash_attention_2 --enable_tp

# passkey retrieval accuracy
python -m main.eval_passkey  --data_root $data_root --model_name_or_path $model --attn_impl flash_attention_2 --enable_tp --chat_template mistral

# needle-in-a-haystack accuracy (remember to set OPENAI_API_KEY in your environmental variable)
python -m main.eval_needle --data_root $data_root --model_name_or_path $model --attn_impl flash_attention_2 --enable_tp --chat_template mistral --gpt_eval

# topic retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_topic --num_topic 5 10 20 30 40 50 60 70 --data_root $data_root --model_name_or_path $model --attn_impl flash_attention_2 --chat_template mistral

# longbench
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --model_name_or_path $model --attn_impl flash_attention_2 --max_length 31500 --chat_template mistral

# infbench
python -m main.eval_infbench --data_root $data_root --model_name_or_path $model --attn_impl flash_attention_2 --chat_template mistral --enable_tp --max_length 128000
```
