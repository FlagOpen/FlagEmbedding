if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="passage"

eval_args="\
    --eval_name msmarco \
    --dataset_dir ./msmarco/data \
    --dataset_names $dataset_names \
    --splits dev \
    --corpus_embd_save_dir ./msmarco/corpus_embd \
    --output_dir ./msmarco/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite True \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./msmarco/msmarco_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
"

model_args="\
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir $HF_HUB_CACHE \
    --reranker_max_length 1024 \
"

cmd="python -m FlagEmbedding.evaluation.msmarco \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
