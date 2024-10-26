if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

dataset_names="bn hi sw"

eval_args="\
    --eval_name miracl \
    --dataset_dir /share/jianlv/evaluation/miracl/data \
    --dataset_names $dataset_names \
    --splits train dev \
    --corpus_embd_save_dir /share/jianlv/data/miracl/corpus_embd \
    --output_dir /share/jianlv/evaluation/miracl/search_results \
    --search_top_k 1000 --rerank_top_k 100 \
    --cache_path $HF_HUB_CACHE \
    --overwrite \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./miracl_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
"

model_args="\
    --embedder_name_or_path BAAI/bge-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir $HF_HUB_CACHE \
"

cmd="python -m FlagEmbedding.evaluation.miracl \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
