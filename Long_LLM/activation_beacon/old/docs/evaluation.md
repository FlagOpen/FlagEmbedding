# Evaluation

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


## Long-Context Generation
### Language Modeling Perplexity
```bash
data_root="/data/activation-beacon"

# NOTE: in the first run, the tokenization could be super slow (often consumes half an hour). However the tokenized corpus will be saved and reused. Be patient.

################ Llama-2 ################
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --use_flash_attention_2

################ PI ################
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --use_flash_attention_2 --rope_method linear --rope_factor 8

################ NTK ################
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --use_flash_attention_2 --rope_method dynamic --rope_factor 2

################ LongLlama ################
# OOM given 32K
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --model_name_or_path syzymon/long_llama_code_7b_instruct
# evaluate 16K instead
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 16384 --model_name_or_path syzymon/long_llama_code_7b_instruct

################ LongChat ################
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --model_name_or_path lmsys/longchat-7b-v1.5-32k --use_flash_attention_2

################ Activation Beacon ################
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat

# evaluating with 400K context (increase stride to 100K so sliding window evaluation is faster)
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 400000 --stride 100000 --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat

# evaluating with 1M context (increase stride to 100K so sliding window evaluation is faster)
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 1000000 --stride 100000 --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat --rope_method dynamic --rope_factor 2
```

By default the perplexity is evaluated on PG19 test set. You can evaluate on Proof-Pile and CodeParrot by specifying `eval_data`:
```bash
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat --eval_data activation-beacon:lm/proof-pile.json
torchrun --nproc_per_node 8 -m main.eval_lm --data_root $data_root --max_length 32768 --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat --eval_data activation-beacon:lm/codeparrot.json
```

The results can be found at `data/results/lm/pg19.log`.


## Long-Context Understanding
### LongBench

```bash
data_root="/data/activation-beacon"

################ Llama-2 ################
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 3500 --use_flash_attention_2

################ PI ################
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 15500 --use_flash_attention_2 --rope_method linear --rope_factor 4

################ NTK ################
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 15500 --use_flash_attention_2 --rope_method dynamic --rope_factor 2

################ LongLlama ################
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 15500 --model_name_or_path syzymon/long_llama_code_7b_instruct

################ LongChat ################
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 31500 --model_name_or_path lmsys/longchat-7b-v1.5-32k --use_flash_attention_2

################ Activation Beacon ################
torchrun --nproc_per_node 8 -m main.eval_longbench --data_root $data_root --max_length 15500 --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat
```

The results can be found at `data/results/longbench/metrics.log`.


### Topic Retrieval
```bash
data_root="/data/activation-beacon"

################ Llama-2 ################
torchrun --nproc_per_node 8 -m main.eval_longeval --data_root $data_root --use_flash_attention_2

################ PI ################
torchrun --nproc_per_node 8 -m main.eval_longeval --data_root $data_root --use_flash_attention_2 --rope_method linear --rope_factor 4

################ NTK ################
torchrun --nproc_per_node 8 -m main.eval_longeval --data_root $data_root --use_flash_attention_2 --rope_method dynamic --rope_factor 2

################ LongLlama ################
torchrun --nproc_per_node 8 -m main.eval_longeval --data_root $data_root --model_name_or_path syzymon/long_llama_code_7b_instruct

################ LongChat ################
torchrun --nproc_per_node 8 -m main.eval_longeval --data_root $data_root --model_name_or_path lmsys/longchat-7b-v1.5-32k --use_flash_attention_2

################ Activation Beacon ################
torchrun --nproc_per_node 8 -m main.eval_longeval --data_root $data_root --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat
```

The results can be found at `data/results/longeval/topic_retrieval.log`.


### Passkey Retrieval
```bash
data_root="/data/activation-beacon"

################ Llama-2 ################
python -m main.eval_passkey --data_root $data_root --use_flash_attention_2

################ PI ################
python -m main.eval_passkey --data_root $data_root --use_flash_attention_2 --rope_method linear --rope_factor 4

################ NTK ################
python -m main.eval_passkey --data_root $data_root --use_flash_attention_2 --rope_method dynamic --rope_factor 2

################ LongLlama ################
python -m main.eval_passkey --data_root $data_root --model_name_or_path syzymon/long_llama_code_7b_instruct

################ LongChat ################
python -m main.eval_passkey --data_root $data_root --model_name_or_path lmsys/longchat-7b-v1.5-32k --use_flash_attention_2

################ Activation Beacon ################
python -m main.eval_passkey --data_root $data_root --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat

# enable retrieval to improve memory accuracy (still in progress)
python -m main.eval_passkey --data_root $data_root --model_name_or_path namespace-Pt/activation-beacon-llama2-7b-chat --enable_beacon --retrieval_method bm25 --retrieval_topk 3 --beacon_ratio 2 128
```

The results can be found at `data/results/passkey/metrics.log`.
