if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="fiqa arguana cqadupstack"

eval_args="\
    --eval_name beir \
    --dataset_dir ./beir/data \
    --dataset_names $dataset_names \
    --splits test dev \
    --corpus_embd_save_dir ./beir/corpus_embd \
    --output_dir ./beir/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./beir/beir_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --ignore_identical_ids True \
"

model_args="\
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir $HF_MODEL_CACHE \
    --reranker_max_length 1024 \
"

cmd="python -m FlagEmbedding.evaluation.beir \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
