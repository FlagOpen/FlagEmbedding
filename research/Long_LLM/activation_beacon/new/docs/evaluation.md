# Evaluation

Make sure you have created the environment and downloaded the data according to [README](../README.md).


```bash
conda activate beacon

model=namespace-Pt/beacon-qwen-2-7b-instruct

# language modeling perplexity
torchrun --nproc_per_node 8 -m main.eval_lm --max_length 100000 --stride 32768 --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024

# passkey retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_passkey --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024

# needle-in-a-haystack accuracy
OPENAI_API_KEY="<you_api_key>" torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024 --gpt_eval

# topic retrieval accuracy
torchrun --nproc_per_node 8 -m main.eval_topic --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024

# longbench
torchrun --nproc_per_node 8 -m main.eval_longbench --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024

# infinitebench
torchrun --nproc_per_node 8 -m main.eval_infbench --model_name_or_path $model --enable_beacon --beacon_ratio_mix adapt-1024
```

All evaluation results will be saved at `data/results`.
