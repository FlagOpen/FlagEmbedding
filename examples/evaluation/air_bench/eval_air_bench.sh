if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

eval_args="\
    --benchmark_version AIR-Bench_24.05 \
    --task_types qa long-doc \
    --domains arxiv \
    --languages en \
    --splits dev test \
    --output_dir ./air_bench/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_dir $HF_HUB_CACHE \
    --overwrite False \
"

model_args="\
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --model_cache_dir $HF_HUB_CACHE \
    --reranker_max_length 1024 \
"

cmd="python -m FlagEmbedding.evaluation.air_bench \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
