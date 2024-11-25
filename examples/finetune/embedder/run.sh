torchrun --nproc_per_node 1 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path BAAI/bge-large-en-v1.5 \
    --train_data ./my_data/finetune_data_submission_minedHN.jsonl \
    --temperature 0.02 \
    --output_dir ./FT-1125-bge-large-en-v1.5-submission \
    --save_steps 250 \
    --per_device_train_batch_size 4 \
    --logging_steps 50 \
    --query_max_len 512 \
    --passage_max_len 64 \
    --train_group_size 8 \
    --cache_dir ./cache/model \
    --cache_path ./cache/data \
    --query_max_len 512 \
    --passage_max_len 64 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --negatives_cross_device