if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="en zh_cn"

eval_args="\
    --eval_name mkqa \
    --dataset_dir ./mkqa/data \
    --dataset_names $dataset_names \
    --splits test \
    --corpus_embd_save_dir ./mkqa/corpus_embd \
    --output_dir ./mkqa/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 20 \
    --eval_output_method markdown \
    --eval_output_path ./mkqa/mkqa_eval_results.md \
    --eval_metrics qa_recall_at_20 \
"

model_args="\
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir $HF_HUB_CACHE \
    --reranker_max_length 1024 \
"

cmd="python -m FlagEmbedding.evaluation.mkqa \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
