#!/bin/bash

LOG_FILE="logs/training_20241128_015203.log"
START_EPOCH=7
TOTAL_EPOCHS=8
OUTPUT_DIR="FT-1125-bge-large-en-v1.5-validation-v3"

# Clean up old checkpoints - keep only the latest one
CHECKPOINTS=($(ls -d $OUTPUT_DIR/checkpoint-*/ | sort -V))
if [ ${#CHECKPOINTS[@]} -gt 1 ]; then
    echo "Removing older checkpoint: ${CHECKPOINTS[0]}"
    rm -rf "${CHECKPOINTS[0]}"
fi