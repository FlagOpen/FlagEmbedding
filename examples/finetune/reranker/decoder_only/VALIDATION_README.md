# Reranker Fine-tuning with Validation

This guide explains how to use validation during reranker fine-tuning.

## Overview

Validation support has been added to the reranker fine-tuning pipeline, allowing you to monitor model performance on a held-out evaluation set during training. This helps with:
- Early stopping detection
- Hyperparameter tuning
- Overfitting detection
- Model selection

## Quick Start

Use the `base_with_eval.sh` script which includes validation configuration:

```bash
bash base_with_eval.sh
```

## Data Format

Evaluation data uses the same format as training data:

```jsonl
{"query": "What is the capital of France?", "pos": ["Paris is the capital of France."], "neg": ["London is the capital of England.", "Berlin is the capital of Germany."]}
{"query": "Who wrote Romeo and Juliet?", "pos": ["William Shakespeare wrote Romeo and Juliet."], "neg": ["Charles Dickens wrote Oliver Twist.", "Jane Austen wrote Pride and Prejudice."]}
```

**Important Notes:**
- First passage in `pos` will be used as the positive example
- Negative passages will be sampled from `neg` list
- Each query should have at least `eval_group_size - 1` negative passages
- No knowledge distillation scores (`pos_scores`, `neg_scores`) are used during evaluation

## Configuration Parameters

### Data Arguments

```bash
--eval_data <path>              # Path(s) to evaluation data files
--eval_group_size <int>         # Number of passages per query (default: 8)
                                # Must be: 1 positive + (N-1) negatives
```

### Training Arguments

```bash
--do_eval                       # Enable evaluation during training
--eval_strategy <strategy>      # When to evaluate: "no", "steps", or "epoch"
                                # - "no": No evaluation (default)
                                # - "steps": Evaluate every eval_steps
                                # - "epoch": Evaluate at end of each epoch
--eval_steps <int>              # Evaluate every N steps (when eval_strategy="steps")
--eval_on_start                 # Run evaluation before training starts
--per_device_eval_batch_size <int>  # Batch size for evaluation (default: 8)
```

## Evaluation Metrics

The following IR (Information Retrieval) metrics are computed automatically:

| Metric | Description |
|--------|-------------|
| `loss` | Cross-entropy loss for ranking (lower is better) |
| `accuracy` | Percentage of queries where positive passage ranks #1 |
| `mrr` | Mean Reciprocal Rank of the positive passage |
| `recall@k` | Percentage of queries where positive is in top-k (k=1,3,5,10,20) |
| `ndcg@k` | Normalized Discounted Cumulative Gain at k (k=1,3,5,10,20) |
| `mean_score` | Average relevance score across all passages |
| `mean_positive_score` | Average score for positive passages |
| `mean_negative_score` | Average score for negative passages |

## Example Configurations

### Evaluate Every Epoch

```bash
training_args="\
    --do_eval \
    --eval_strategy epoch \
    --per_device_eval_batch_size 4 \
"
```

### Evaluate Every 100 Steps

```bash
training_args="\
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 4 \
"
```

### Evaluate at Start and Every Epoch

```bash
training_args="\
    --do_eval \
    --eval_strategy epoch \
    --eval_on_start \
    --per_device_eval_batch_size 4 \
"
```

## Full Example

```bash
export WANDB_MODE=disabled

# Training data
train_data="./data/train.jsonl"

# Evaluation data (separate from training)
eval_data="./data/eval.jsonl"

# Model arguments
model_args="\
    --model_name_or_path BAAI/bge-reranker-v2-gemma \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --model_type decoder \
"

# Data arguments
data_args="\
    --train_data $train_data \
    --eval_data $eval_data \
    --train_group_size 8 \
    --eval_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
"

# Training arguments with evaluation
training_args="\
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-4 \
    --do_eval \
    --eval_strategy epoch \
    --logging_steps 10 \
    --save_steps 500 \
"

# Run training
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.reranker.decoder_only.base \
    $model_args \
    $data_args \
    $training_args
```

## Monitoring Results

Evaluation metrics will be logged to:
1. Console output during training
2. TensorBoard logs (if enabled)
3. WandB (if enabled and `WANDB_MODE` is not disabled)
4. `trainer_state.json` in the output directory

Example console output:
```
{'eval_loss': 0.234, 'eval_accuracy': 0.856, 'eval_mrr': 0.912, 'eval_recall@1': 0.856, 'eval_recall@3': 0.943, 'eval_recall@5': 0.967, 'eval_recall@10': 0.989, 'eval_recall@20': 0.995, 'eval_ndcg@1': 0.856, 'eval_ndcg@3': 0.897, 'eval_ndcg@5': 0.908, 'eval_ndcg@10': 0.915, 'eval_ndcg@20': 0.918, 'eval_mean_score': 2.341, 'eval_mean_positive_score': 4.123, 'eval_mean_negative_score': 1.987, 'eval_runtime': 12.34, 'eval_samples_per_second': 81.2, 'eval_steps_per_second': 2.43, 'epoch': 1.0}
```

## Tips

1. **Separate train/eval data**: Use different data for evaluation to detect overfitting
2. **Group size**: Set `eval_group_size` to match your use case (e.g., 100 for reranking top-100 candidates)
3. **Batch size**: Use larger `per_device_eval_batch_size` than training batch size for faster evaluation
4. **Evaluation frequency**: Balance between monitoring frequency and training speed
   - For small datasets: Use `eval_strategy=epoch`
   - For large datasets: Use `eval_strategy=steps` with appropriate `eval_steps`

## Compatibility

- Compatible with `transformers==4.44.2`
- Works with all decoder-only reranker models (e.g., bge-reranker-v2-gemma)
- Supports LoRA and full fine-tuning
- Compatible with DeepSpeed distributed training
