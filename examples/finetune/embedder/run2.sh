#!/bin/bash
# validation v2: 
set -e
mkdir -p logs
# Generate timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
ERROR_LOG="logs/error_${TIMESTAMP}.log"

# Redirect all output to both terminal and log files using tee
exec 1> >(tee -a "${LOG_FILE}") 2> >(tee -a "${ERROR_LOG}")

echo "Starting script at $(date)"
echo "Logging to $LOG_FILE"
echo "Error logging to $ERROR_LOG"


OUTPUT_DIR="./FT-1125-bge-large-en-v1.5-validation-v4"
NUM_EPOCHS=7
    
# Check if there is a checkpoint file
if [ -d "$OUTPUT_DIR" ] && ls "$OUTPUT_DIR"/checkpoint-*/trainer_state.json 1>/dev/null 2>&1; then
    RESUME_CHECKPOINT_ARG="--resume_from_checkpoint True"
else
    RESUME_CHECKPOINT_ARG=""
fi

# Training
torchrun --nproc_per_node 1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --train_data ./bge_finetune_data/finetune_data_validation_minedHN.jsonl \
    --corpus_path ./eval_data/corpus.jsonl \
    --eval_data ./eval_data/queries_v2.jsonl \
    --num_train_epochs $NUM_EPOCHS \
    $RESUME_CHECKPOINT_ARG \
    --save_total_limit 2 \
    --temperature 0.03 \
    --output_dir $OUTPUT_DIR \
    --save_steps 1000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --logging_steps 50 \
    --query_max_len 512 \
    --passage_max_len 128 \
    --train_group_size 12 \
    --cache_dir ./cache/model \
    --cache_path ./cache/data \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --negatives_cross_device || exit 1