#!/bin/bash


OUTPUT_DIR="./FT-1125-bge-large-en-v1.5-validation-v4"

if [ -d "$OUTPUT_DIR" ] && ls "$OUTPUT_DIR"/checkpoint-*/trainer_state.json 1>/dev/null 2>&1; then
    RESUME_CHECKPOINT_ARG="--resume_from_checkpoint "
else
    RESUME_CHECKPOINT_ARG=""
fi

echo $RESUME_CHECKPOINT_ARG



python -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path BAAI/bge-small-en-v1.5 \
    --train_data ./bge_finetune_data/finetune_data_validation_minedHN.jsonl \
    --corpus_path ./eval_data/corpus.jsonl \
    --eval_data ./eval_data/queries_v2.jsonl \
    --output_dir ./debug_output \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --max_steps 10 \
    --logging_steps 1