#!/bin/bash
# Set root path
ROOT=/home

# Set training machines
# For more details, refer to https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node
HOST_FILE_CONTENT="\
localhost slots=8\n\
"
HOST_FILE=hostfile
printf "$HOST_FILE_CONTENT" > $HOST_FILE

DISTRIBUTED_ARGS="--hostfile $HOST_FILE"

export LAUNCHER="deepspeed \
    $DISTRIBUTED_ARGS \
	"
# Set cache directory
CACHE_PATH=$ROOT/datasets/.cache

# Set path of deepspeed config file
# For more details, refer to https://huggingface.co/docs/transformers/main_classes/deepspeed#zero
DS_CONFIG_FILE=$ROOT/train/ds_config.json

# Set group size of training
GROUP_SIZE=2

# Set paths of training data. Every path **must be a directory path**.
DATA_PATH="
$ROOT/datasets/toy_train_data \
"

# Set default batch size for training.
# If you want to use effient batching strategy, you should use the script `split_data_by_length.py` to split your data by sequence length firstly. Then the batch size for every batch will depend on its sequence length range, such as len-0-500: 48, len-500-1000: 32, etc., which are defined in `get_file_batch_size()` in `BGE_M3/data.py`.
DEFAULT_BATCH_SIZE=1

# Set number of training epochs.
# For more details, refer to https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
EPOCHS=5
MAX_STEPS=-1

# Set base model and save path
BASE_MODEL=BAAI/bge-m3

SAVE_PATH=$ROOT/models/bge-m3_finetuned
mkdir -p $SAVE_PATH

# Set learning rate
LEARNING_RATE=5e-6

full_options="
  --knowledge_distillation True \
  --output_dir $SAVE_PATH \
  --model_name_or_path $BASE_MODEL \
  --normlized True \
  --temperature 0.02 \
  --do_train  \
  --train_data $DATA_PATH \
  --cache_path $CACHE_PATH \
  --per_device_train_batch_size $DEFAULT_BATCH_SIZE \
  --query_max_len 512 \
  --passage_max_len 8192 \
  --small_threshold 200 \
  --drop_threshold 200 \
  --fp16  \
  --save_steps 1500 \
  --train_group_size $GROUP_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $EPOCHS \
  --max_steps $MAX_STEPS \
  --negatives_cross_device False \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --overwrite_output_dir True \
  --gradient_checkpointing \
  --sentence_pooling_method cls \
  --same_task_within_batch True \
  --shuffle_ratio 0.002 \
  --enable_sub_batch True \
  --deepspeed ${DS_CONFIG_FILE} \
  --unified_finetuning True \
  --use_self_distill True
  "

run_cmd="$LAUNCHER --module FlagEmbedding.BGE_M3.run ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $SAVE_PATH/output.log

set +x
