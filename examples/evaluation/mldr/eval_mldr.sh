if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="hi"

eval_args="\
    --eval_name mldr \
    --dataset_dir ./mldr/data \
    --dataset_names $dataset_names \
    --splits test \
    --corpus_embd_save_dir ./mldr/corpus_embd \
    --output_dir ./mldr/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./mldr/mldr_eval_results.md \
    --eval_metrics ndcg_at_10 \
"

model_args="\
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir $HF_HUB_CACHE \
    --embedder_passage_max_length 8192 \
    --reranker_max_length 8192 \
"

cmd="python -m FlagEmbedding.evaluation.mldr \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
