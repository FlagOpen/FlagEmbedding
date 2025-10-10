if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

# full datasets
# dataset_names="biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems"

# small datasets for quick test
dataset_names="pony theoremqa_theorems"

model_args="\
    --embedder_name_or_path BAAI/bge-reasoner-embed-qwen3-8b-0923 \
    --embedder_model_class decoder-only-base \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --pooling_method last_token \
    --devices cuda:0 cuda:1 \
    --cache_dir $HF_HUB_CACHE \
    --embedder_batch_size 2 \
    --embedder_query_max_length 8192 \
    --embedder_passage_max_length 8192 \
"

split_list=("examples" "gpt4_reason")

for split in "${split_list[@]}"; do
    eval_args="\
        --task_type short \
        --use_special_instructions True \
        --eval_name bright_short \
        --dataset_dir ./bright_short/data \
        --dataset_names $dataset_names \
        --splits $split \
        --corpus_embd_save_dir ./bright_short/corpus_embd \
        --output_dir ./bright_short/search_results/$split \
        --search_top_k 2000 \
        --cache_path $HF_HUB_CACHE \
        --overwrite False \
        --k_values 1 10 100 \
        --eval_output_method markdown \
        --eval_output_path ./bright_short/eval_results_$split.md \
        --eval_metrics ndcg_at_10 recall_at_10 recall_at_100 \
    "

    cmd="python -m FlagEmbedding.evaluation.bright \
        $eval_args \
        $model_args \
    "

    echo $cmd
    eval $cmd

done
