export WANDB_MODE=disabled

train_data="\
    ../example_data/prompt_based/examples.jsonl "

# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=2
gradient_accumulation_steps=1
train_group_size=8

# set num_gpus to 2 for testing
num_gpus=2

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path BAAI/bge-reranker-v2-gemma \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --use_flash_attn True \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_merged_lora_model True \
    --model_type decoder \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
"

training_args="\
    --output_dir ./test_decoder_only_base_bge-reranker-v2-gemma \
    --overwrite_output_dir \
    --learning_rate 2e-4 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ../../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd