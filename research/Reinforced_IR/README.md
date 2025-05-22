<div align="center">
<h1> Reinforced IR: A Reinforcement Framework For Cross-Domain Information Retrieval [<a href="https://arxiv.org/abs/2502.11562v1">paper</a>]</h1>
</div>



We propose a domain adaptation framework called **Reinforced-IR**, which jointly adapts the retriever and LLM-based generator using a unlabeled corpus. Our method is distinguished for its design of **self-boosting** algorithm. It starts with a list of pseudo questions generated from the target domain’s unlabeled corpus. 

**Reinforcement Learning of generator with Retriever’s Feedback (RLRF):** The LLM-based generator is reinforced to perform high-quality query augmentation using the retriever’s feedback, such that relevant documents can be optimally retrieved for downstream tasks.

**Reinforcement Learning of retriever with Generator’s Feedback (RLGF):** The retriever is reinforced to discriminate the relevant documents preferred by the LLM-based generator. 

With the alternating execution of these two operations, the end-to-end retrieval performance can be progressively enhanced for the target domain.

## Environment

You can install the environment by:

```bash
conda create -n reinforced_ir python=3.10
conda activate reinforced_ir

# prepare torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# prepare usage environment
pip install -r requirements.txt
pip install transformers==4.46.0

# prepare training environment
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

# prepare evaluation environment
pip install pytrec_eval
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Synthetic data

| Data                                                         | Introduction                                 |
| ------------------------------------------------------------ | -------------------------------------------- |
| [cfli/Reinforced-IR-synthetic](https://huggingface.co/datasets/cfli/Reinforced-IR-synthetic) | The synthetic data we used in our experiment |

### Usage without Training

You can use the model with the following code:

```bash
cd ./inference
python
```

If you don't finetune the retriever and the generator, you can use directly:

```python
from ir_model import Reinforced_IR_Model

api_key=''
base_url=''

model = Reinforced_IR_Model(
    model_name_or_path='BAAI/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    use_fp16=True,
    devices=['cuda:0'],
    generator_model_name_or_path='gpt-4o-mini',
    api_key=api_key,
    base_url=base_url,
    temperature=0,
    model_type='gpt' # gpt, llm, llm_instruct
)

queries = ["how much protein should a female eat", "summit define"]

documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

task_instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
answer_type = 'passage'

embeddings_1 = model.encode_queries(
    task_instruction=task_instruction,
    answer_type=answer_type,
    queries=queries
)
embeddings_2 = model.encode_corpus(corpus=documents)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

## Train from scratch

### Prepare Data

**BEIR:** For BEIR dataset, you can get corpus from the following link: https://huggingface.co/datasets/BeIR/beir#data-splits

**AIR-Bench:** For AIR-Bench dataset, you can get corpus from the following link: https://huggingface.co/AIR-Bench

For example, for msmarco dataset, you can save with the following format:

```python
[
    "Clinical features of culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia OBJECTIVE: This retrospective chart review describes the epidemiology and clinical features of 40 patients with culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia. METHODS: Patients with positive M. pneumoniae cultures from respiratory specimens from January 1997 through December 1998 were identified through the Microbiology records. Charts of patients were reviewed. RESULTS: 40 patients were identified, 33 (82.5%) of whom required admission. Most infections (92.5%) were community-acquired. The infection affected all age groups but was most common in infants (32.5%) and pre-school children (22.5%). It occurred year-round but was most common in the fall (35%) and spring (30%). More than three-quarters of patients (77.5%) had comorbidities. Twenty-four isolates (60%) were associated with pneumonia, 14 (35%) with upper respiratory tract infections, and 2 (5%) with bronchiolitis. Cough (82.5%), fever (75%), and malaise (58.8%) were the most common symptoms, and crepitations (60%), and wheezes (40%) were the most common signs. Most patients with pneumonia had crepitations (79.2%) but only 25% had bronchial breathing. Immunocompromised patients were more likely than non-immunocompromised patients to present with pneumonia (8/9 versus 16/31, P = 0.05). Of the 24 patients with pneumonia, 14 (58.3%) had uneventful recovery, 4 (16.7%) recovered following some complications, 3 (12.5%) died because of M pneumoniae infection, and 3 (12.5%) died due to underlying comorbidities. The 3 patients who died of M pneumoniae pneumonia had other comorbidities. CONCLUSION: our results were similar to published data except for the finding that infections were more common in infants and preschool children and that the mortality rate of pneumonia in patients with comorbidities was high.", 
    "Nitric oxide: a pro-inflammatory mediator in lung disease? Inflammatory diseases of the respiratory tract are commonly associated with elevated production of nitric oxide (NO\u2022) and increased indices of NO\u2022 -dependent oxidative stress. Although NO\u2022 is known to have anti-microbial, anti-inflammatory and anti-oxidant properties, various lines of evidence support the contribution of NO\u2022 to lung injury in several disease models. On the basis of biochemical evidence, it is often presumed that such NO\u2022 -dependent oxidations are due to the formation of the oxidant peroxynitrite, although alternative mechanisms involving the phagocyte-derived heme proteins myeloperoxidase and eosinophil peroxidase might be operative during conditions of inflammation. Because of the overwhelming literature on NO\u2022 generation and activities in the respiratory tract, it would be beyond the scope of this commentary to review this area comprehensively. Instead, it focuses on recent evidence and concepts of the presumed contribution of NO\u2022 to inflammatory diseases of the lung."]
```

For all data, you can save with the following format:

```
├─data
|  ├─msmarco
|    ├─corpus.json
|  ├─trec-covid
|    ├─corpus.json
|  ├─nq
|    ├─corpus.json
|  ├...
```

### Finetuning

Taking one round as example:

#### Synthetic Query

You can get our synthetic queries from the following link: [cfli/Reinforced-IR-synthetic](https://huggingface.co/datasets/cfli/Reinforced-IR-synthetic)

Or you can generate synthetic queries yourself:

```bash
cd ./data_generation
api_key='sk-gzAdunPMOSEDdotUkMgwnHKN5eP4a2vZx8GKBeN1hHH017z0'
base_url='https://api.xiaoai.plus/v1'
dataset_name='msmarco'
python generate_universal_query.py \
    --generate_model_path gpt-4o-mini \
    --api_key ${api_key} \
    --base_url ${base_url} \
    --temperature 0.5 \
    --top_p 1.0 \
    --max_tokens 4096 \
    --model_type gpt \
    --train_num 15000 \
    --dataset_path ../data \
    --output_dir ../synthetic/generator_round1 \
    --dataset_name ${dataset_name}
```

For all data, it will be saved with the following format:

```
├─inference
├─finetune
├─data_generation
├─data
|  ├─msmarco
|    ├─corpus.json
|  ├...
├─synthetic
|  ├─generator_round1
|    ├─msmarco
|      ├─queries.json
|    ├...
|  ├─retriever_round1
|    ├─msmarco
|      ├─queries.json
|    ├...
```

#### Prepare Data for Generator

You can generate data with the following command:

```bash
cd ./data_generation
dataset_name='msmarco'
python generate_generator_data.py \
    --generate_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --gpu_memory_utilization 0.8 \
    --temperature 0.9 \
    --top_p 0.9 \
    --max_tokens 300 \
    --model_type llm_instruct \
    --retrieval_model_name facebook/contriever \
    --pooling_method mean \
    --retrieval_query_prompt "" \
    --max_length 512 \
    --batch_size 512 \
    --dataset_path ../data \
    --output_dir ../synthetic/generator_round1 \
    --dpo_num 5 \
    --threshold 0.95 \
    --normalize_embeddings False \
    --dataset_name ${dataset_name}

cd ../finetune/generator
python update_file.py \
    --input_path ../../synthetic/generator_round1/${dataset_name}/train.jsonl \
    --output_path ../../finetune/generator/data/train_llamafactory.jsonl \
    --file_name ${dataset_name} \
    --info_path ../../finetune/generator/data/dataset_info.json
```

For all data, it will be saved with the following format:

```
├─inference
├─finetune
|  ├─generator
|    ├─data
|      ├─train_llamafactory.jsonl
|      ├─dataset_info.json
|  ├...
├─data_generation
├─data
|  ├─msmarco
|    ├─corpus.json
|  ├...
├─synthetic
|  ├─generator_round1
|    ├─msmarco
|      ├─queries.json
|      ├─answers.json
|      ├─train.jsonl
|    ├...
|  ├─retriever_round1
|    ├─msmarco
|      ├─queries.json
|    ├...
```

#### Training Generator

```bash
dataset_name='msmarco'
cd ./finetune/generator

# Execute the training command
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled llamafactory-cli train \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ../../finetune_result/${dataset_name}/generator_round1 \
    --stage dpo \
    --do_train true \
    --lora_alpha 64 \
    --lora_rank 32 \
    --finetuning_type lora \
    --lora_target all \
    --pref_beta 0.1 \
    --pref_loss sigmoid \
    --deepspeed ../stage1.json \
    --dataset ${dataset_name} \
    --cutoff_len 4096 \
    --vllm_maxlen 4096 \
    --max_samples 999999999999999 \
    --overwrite_cache false \
    --preprocessing_num_workers 32 \
    --logging_steps 1 \
    --save_steps 3000 \
    --plot_loss true \
    --overwrite_output_dir false \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --do_eval false \
    --val_size 0.05 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --max_new_tokens 512 \
    --template llama3

llamafactory-cli export \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ../../finetune_result/${dataset_name}/generator_round1 \
    --template llama3 \
    --finetuning_type lora \
    --export_size 2 \
    --export_dir ../../finetune_result/${dataset_name}/generator_round1/merged_model \
    --export_device cpu \
    --export_legacy_format false

python save_tokenizer.py \
	--model_path meta-llama/Meta-Llama-3-8B-Instruct \
	--output_path ../../finetune_result/${dataset_name}/generator_round1/merged_model
```

#### Prepare Data for Retriever

You can generate data with the following command:

```bash
cd ./data_generation
dataset_name='msmarco'
python generate_retriever_data.py \
    --generate_model_path ../finetune_result/${dataset_name}/generator_round1/merged_model \
    --gpu_memory_utilization 0.8 \
    --temperature 0 \
    --top_p 1.0 \
    --max_tokens 300 \
    --model_type llm_instruct \
    --retrieval_model_name BAAI/bge-large-en-v1.5 \
    --pooling_method cls \
    --retrieval_query_prompt "Represent this sentence for searching relevant passages: " \
    --max_length 512 \
    --batch_size 512 \
    --dataset_path ../data \
    --output_dir ../synthetic/retriever_round1 \
    --normalize_embeddings True \
    --dataset_name ${dataset_name}
```

If you want to get score for distill, you can process data with the following command:

```bash
cd ./data_generation
dataset_name='msmarco'
python generate_retriever_distill_data.py \
    --generate_model_path ../finetune_result/${dataset_name}/generator_round1/merged_model \
    --gpu_memory_utilization 0.8 \
    --temperature 0 \
    --top_p 1.0 \
    --max_tokens 300 \
    --model_type llm_instruct \
    --dataset_path ../data \
    --output_dir ../synthetic/retriever_round1 \
    --dataset_name ${dataset_name}
```

For all data, it will be saved with the following format:

```
├─inference
├─finetune
|  ├─generator
|    ├─data
|      ├─train_llamafactory.jsonl
|      ├─dataset_info.json
|  ├...
├─data_generation
├─data
|  ├─msmarco
|    ├─corpus.json
|  ├...
├─synthetic
|  ├─generator_round1
|    ├─msmarco
|      ├─queries.json
|      ├─answers.json
|      ├─train.jsonl
|    ├...
|  ├─retriever_round1
|    ├─msmarco
|      ├─queries.json
|      ├─answers.json
|      ├─train.jsonl
|    ├...
```

#### Training Retriever

```bash
#!/bin/bash

cd ./finetune/retriever
dataset_name='msmarco'

torchrun --nproc_per_node 8 \
    -m run \
    --output_dir ../../finetune_result/${dataset_name}/retriever_round1 \
    --model_name_or_path facebook/contriever \
    --train_data ../../synthetic/retriever_round1/msmarco/train.jsonl \
    --same_dataset_within_batch \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --dataloader_drop_last True \
    --normalize_embeddings False \
    --temperature 0.02 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --gradient_checkpointing \
    --deepspeed ../stage1.json \
    --train_group_size 16 \
    --negatives_cross_device \
    --logging_steps 1 \
    --save_steps 1000 \
    --query_instruction_for_retrieval '' \
    --sentence_pooling_method mean \
    --knowledge_distillation True \
    --kd_loss_type m3_kd_loss \
    --training_type retrieval_answer
```

### Inference

You can use the model with the following code:

```bash
cd ./inference
python
```

And then, run with python:

```python
from ir_model import Reinforced_IR_Model

dataset_name='msmarco'
model = Reinforced_IR_Model(
    model_name_or_path=f'../finetune_result/{dataset_name}/retriever_round1',
    query_instruction_for_retrieval="",
    pooling_method='mean',
    normalize_embeddings=False,
    model_class='encoder-only-base',
    use_fp16=True,
    devices=['cuda:0'],
    generator_model_name_or_path=f'../finetune_result/{dataset_name}/generator_round1/merged_model',
    temperature=0,
    model_type='llm_instruct' # gpt, llm, llm_instruct
)

queries = ["how much protein should a female eat", "summit define"]

documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

task_instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
answer_type = 'passage'

embeddings_1 = model.encode_queries(
    task_instruction=task_instruction,
    answer_type=answer_type,
    queries=queries
)
embeddings_2 = model.encode_corpus(corpus=documents)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```