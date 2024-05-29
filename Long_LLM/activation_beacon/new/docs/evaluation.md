# Evaluation

## Prerequisite

Make sure you have created the environment and downloaded the data according to [README](../README.md).


## Evaluating Beacon Models
```bash
conda activate beacon

model=namespace-Pt/activation-beacon-mistral-7b

# language modeling perplexity
torchrun --nproc_per_node 8 -m main.eval_lm --max_length 100000 --stride 32768 --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024

# passkey retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_passkey --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024 --chat_template mistral

# needle-in-a-haystack accuracy
OPENAI_API_KEY="<you_api_key>" torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024 --chat_template mistral --gpt_eval

# topic retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_topic --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024 --chat_template mistral

# longbench
torchrun --nproc_per_node 8 -m main.eval_longbench --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024 --chat_template mistral

# infinitebench
torchrun --nproc_per_node 8 -m main.eval_infbench --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024 --chat_template mistral
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

model=mistralai/Mistral-7B-Instruct-v0.2

# language modeling perplexity
python -m main.eval_lm  --max_length 100000 --stride 32768 --model_name_or_path $model --attn_impl flash_attention_2 --enable_tp

# passkey retrieval accuracy
python -m main.eval_passkey  --model_name_or_path $model --attn_impl flash_attention_2 --enable_tp --chat_template mistral

# needle-in-a-haystack accuracy
OPENAI_API_KEY="<you_api_key>" python -m main.eval_needle --model_name_or_path $model --attn_impl flash_attention_2 --enable_tp --chat_template mistral --gpt_eval

# topic retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_topic --model_name_or_path $model --attn_impl flash_attention_2 --chat_template mistral

# longbench
torchrun --nproc_per_node 8 -m main.eval_longbench --model_name_or_path $model --attn_impl flash_attention_2 --chat_template mistral

# infbench
python -m main.eval_infbench --model_name_or_path $model --attn_impl flash_attention_2 --chat_template mistral --enable_tp
```
