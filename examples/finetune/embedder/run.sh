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



OUTPUT_DIR="./FT-1125-bge-large-en-v1.5-validation-v3"
START_EPOCH=7  # Set this to your desired starting epoch
epoch=$START_EPOCH
NUM_EPOCHS=2
TOTAL_EPOCHS=$((START_EPOCH + NUM_EPOCHS - 1))

PREV_MAP=0

while [ $epoch -le $TOTAL_EPOCHS ]; do
    echo "Starting epoch $epoch/$TOTAL_EPOCHS"
    
    # Get latest checkpoint if not first epoch
    if [ $epoch -eq 1 ]; then
        RESUME_CHECKPOINT_ARG=""
    else
        LATEST_CHECKPOINT=$(ls -d $OUTPUT_DIR/checkpoint-*/ | sort -V | tail -n1)
        RESUME_CHECKPOINT_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
    fi
    
    # Training
    torchrun --nproc_per_node 1 \
        -m FlagEmbedding.finetune.embedder.encoder_only.base \
        --model_name_or_path BAAI/bge-large-en-v1.5 \
        --train_data ./my_data/finetune_data_validation_minedHN.jsonl \
        --num_train_epochs $epoch \
        $RESUME_CHECKPOINT_ARG \
        --temperature 0.03 \
        --output_dir $OUTPUT_DIR \
        --save_steps 1000 \
        --per_device_train_batch_size 16 \
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

    echo "Finished training epoch $epoch, starting evaluation"

    # Evaluation after each epoch
    python /root/autodl-tmp/github/kaggle-eedi-math/kernels/bge-instruction/bge-instruction.py \
        --model $OUTPUT_DIR \
        --output_file /root/autodl-tmp/github/kaggle-eedi-math/submission_inst.csv \
        --filter_na_misconception=False \
        --with_instruction=False \
        --query_text_version v1 || exit 1

    echo "Finished evaluation for epoch $epoch"

    # Get the current epoch's MAP metric
    METRICS=$(grep "On validation_v2" "${LOG_FILE}" | tail -n1)
    CURRENT_MAP=$(echo "$METRICS" | grep -o "MAP@25: [0-9.]*" | awk '{print $2}')

    # Compare with previous MAP
    if (( $(echo "$CURRENT_MAP < $PREV_MAP" | bc -l) )); then
        echo "MAP decreased from $PREV_MAP to $CURRENT_MAP. Exiting training."
        break
    fi

    # Update previous MAP
    PREV_MAP=$CURRENT_MAP

    # Clean up old checkpoints - keep only the latest one
    CHECKPOINTS=($(ls -d $OUTPUT_DIR/checkpoint-*/ | sort -V))
    if [ ${#CHECKPOINTS[@]} -gt 1 ]; then
        echo "Removing older checkpoint: ${CHECKPOINTS[0]}"
        rm -rf "${CHECKPOINTS[0]}"
    fi

    # Increment epoch counter
    epoch=$((epoch + 1))
done
echo "Script completed at $(date)"


# Print summary of all epochs' metrics
echo -e "\n=== Training Summary ==="
echo "Metrics for all epochs:"
# Print metrics for epochs START_EPOCH through TOTAL_EPOCHS
LINE_NUM=1
for e in $(seq $START_EPOCH $TOTAL_EPOCHS); do
    echo "epoch $e: $(grep "On validation_v2" "${LOG_FILE}" | sed -n "${LINE_NUM}p")"
    LINE_NUM=$((LINE_NUM + 1))
done
echo "=== End Summary ==="

# === Training Summary ===
# Metrics for all epochs: [train_group_size=12]
# epoch 1: Recall@25: 0.7181, MAP@25: 0.2680
# epoch 2: Recall@25: 0.7449, MAP@25: 0.2762
# epoch 3: Recall@25: 0.7465, MAP@25: 0.2812
# epoch 4: Recall@25: 0.7551, MAP@25: 0.2865
# epoch 5: Recall@25: 0.7540, MAP@25: 0.2868
# epoch 6: Recall@25: 0.7540, MAP@25: 0.2884
# epoch 7: Recall@25: 0.7535, MAP@25: 0.2876
# epoch 8: Recall@25: 0.7567, MAP@25: 0.2913
# === End Summary ===


# # submission v1: 9.5G VRAM
# torchrun --nproc_per_node 1 \
# 	-m FlagEmbedding.finetune.embedder.encoder_only.base \
# 	--model_name_or_path BAAI/bge-large-en-v1.5 \
#     --train_data ./my_data/finetune_data_submission_minedHN.jsonl \
#     --temperature 0.02 \
#     --output_dir ./FT-1125-bge-large-en-v1.5-submission-v1 \
#     --save_steps 250 \
#     --per_device_train_batch_size 4 \
#     --logging_steps 50 \
#     --query_max_len 512 \
#     --passage_max_len 64 \
#     --train_group_size 8 \
#     --cache_dir ./cache/model \
#     --cache_path ./cache/data \
#     --pad_to_multiple_of 8 \
#     --knowledge_distillation False \
#     --overwrite_output_dir \
#     --learning_rate 1e-5 \
#     --fp16 \
#     --num_train_epochs 2 \
#     --dataloader_drop_last True \
#     --warmup_ratio 0.1 \
#     --gradient_checkpointing \
#     --deepspeed ../ds_stage0.json \
#     --negatives_cross_device

