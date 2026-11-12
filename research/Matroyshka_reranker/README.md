<div align="center">
<h1> Fitting Into Any Shape: A Flexible LLM-Based Re-Ranker With Configurable Depth and Width (Matroyshka Re-Ranker) [<a href="https://dl.acm.org/doi/abs/10.1145/3696410.3714620">paper</a>]</h1>
</div>

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. You can get a relevance score by inputting query and passage to the reranker. And the score can be mapped to a float value in [0,1] by sigmoid function.

Here is **Matroyshka Re-Ranker**, which is designed to facilitate **runtime customization** of model layers and sequence lengths at each layer based on users' configurations, it supports flexible lightweight configuration.

The training method have the following features:

- cascaded self-distillation
- factorized compensation

## Environment

You can install the environment by:

```bash
conda create -n reranker python=3.10
conda activate reranker
pip install -r requirements.txt
```

## Model List

| Model                                                        | Introduction                                              |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [BAAI/Matroyshka-ReRanker-passage](https://huggingface.co/BAAI/Matroyshka-ReRanker-passage) | The Matroyshka Re-Ranker fine-tuned on MS MARCO passage   |
| [BAAI/Matroyshka-ReRanker-document](https://huggingface.co/BAAI/Matroyshka-ReRanker-document) | The Matroyshka Re-Ranker fine-tuned on MS MARCO document  |
| [BAAI/Matroyshka-ReRanker-beir](https://huggingface.co/BAAI/Matroyshka-ReRanker-beir) | The Matroyshka Re-Ranker fine-tuned for general retrieval |

### Usage

You can use Matroyshka Re-Ranker with the following code:

```bash
cd ./inference
python
```

And then:

```python
from rank_model import MatroyshkaReranker

compress_ratio = 2 # config your compress ratio
compress_layers = [8, 16] # cofig your layers to compress
cutoff_layers = [20, 24] # config your layers to output

reranker = MatroyshkaReranker(
    model_name_or_path='BAAI/Matroyshka-ReRanker-passage',
    peft_path=[
        './models/Matroyshka-ReRanker-passage/compensate/layer/full'
    ]
    use_fp16=True,
    cache_dir='./model_cache',
    compress_ratio=compress_ratio,
    compress_layers=compress_layers,
    cutoff_layers=cutoff_layers
)

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```

## Fine-tune

### Cascaded Self-distillation

For cascaded self-distillation, you can use the following script:

```bash
cd self_distillation

train_data_path="..."
your_huggingface_token="..."

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./result_self_distillation \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--train_data ${train_data_path} \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--query_max_len 32 \
--passage_max_len 192 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 100 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed /share/chaofan/code/stage/stage1.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--loss_type 'only logits' \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj linear_head \
--token ${your_huggingface_token} \
--cache_dir ../../model_cache \
--cache_path ../../data_cache \
--padding_side right \
--start_layer 4 \
--layer_sep 1 \
--layer_wise True \
--compress_ratios 1 2 4 8 \
--compress_layers 4 8 12 16 20 24 28 \
--train_method distill_fix_layer_teacher
```

- ### Factorized Compensation

For layer compensation, you can use the following script:

```bash
cd finetune/compensation

train_data_path="..."
your_huggingface_token="..."
raw_peft_path="../../self_distillation/result_self_distillation"

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./result_compensation_layer \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--raw_peft ${raw_peft_path} \
--train_data ${train_data_path} \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--query_max_len 32 \
--passage_max_len 192 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 500 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage1.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--loss_type 'only logits' \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj linear_head \
--token ${your_huggingface_token} \
--cache_dir ../../model_cache \
--cache_path ../../data_cache \
--padding_side right \
--start_layer 4 \
--layer_sep 1 \
--layer_wise True \
--compress_ratios 1 \
--compress_layers 4 8 12 16 20 24 28 \
--train_method normal \
--finetune_type layer
```

For token compression, you can use the following script:

```bash
cd finetune/compensation

train_data_path="..."
your_huggingface_token="..."
raw_peft_path="../../self_distillation/result_self_distillation"
compress_ratio=2

torchrun --nproc_per_node 8 \
run.py \
--output_dir ./result_compensation_token_compress_ratio_${compress_ratio} \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--raw_peft ${raw_peft_path} \
--train_data ${train_data_path} \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--query_max_len 32 \
--passage_max_len 192 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 500 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage1.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--loss_type 'only logits' \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj linear_head \
--token ${your_huggingface_token} \
--cache_dir ../../model_cache \
--cache_path ../../data_cache \
--padding_side right \
--start_layer 4 \
--layer_sep 1 \
--layer_wise True \
--compress_ratios ${compress_ratio} \
--compress_layers 4 8 12 16 20 24 28 \
--train_method normal \
--finetune_type token
```

### Inference

You can use self finetuned Matroyshka Re-Ranker with the following code:

```bash
cd ./inference
python
```

And then:

```python
from rank_model import MatroyshkaReranker

compress_ratio = 2 # config your compress ratio
compress_layers = [8, 16] # cofig your layers to compress
cutoff_layers = [20, 24] # config your layers to output

reranker = MatroyshkaReranker(
    model_name_or_path='mistralai/Mistral-7B-v0.1',
    peft_path=[
        './finetune/self_distillation/result_self_distillation',
        './finetune/compensation/result_compensation_token_compress_ratio_2',
    ],
    use_fp16=True,
    cache_dir='./model_cache',
    compress_ratio=compress_ratio,
    compress_layers=compress_layers,
    cutoff_layers=cutoff_layers
)

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```

### 