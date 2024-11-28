#!/bin/bash


OUTPUT_DIR="./FT-1125-bge-large-en-v1.5-validation-v4"

if [ -d "$OUTPUT_DIR" ] && ls "$OUTPUT_DIR"/checkpoint-*/trainer_state.json 1>/dev/null 2>&1; then
    RESUME_CHECKPOINT_ARG="--resume_from_checkpoint "
else
    RESUME_CHECKPOINT_ARG=""
fi

echo $RESUME_CHECKPOINT_ARG