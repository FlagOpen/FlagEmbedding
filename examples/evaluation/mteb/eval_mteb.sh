if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

languages="eng"
tasks="NFCorpus BiorxivClusteringS2S SciDocsRR"

eval_args="\
    --eval_name mteb \
    --output_dir ./mteb/search_results \
    --languages $languages \
    --tasks $tasks \
    --eval_output_path ./mteb/mteb_eval_results.json
"

model_args="\
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --devices cuda:7 \
    --cache_dir $HF_HUB_CACHE \
"

cmd="python -m FlagEmbedding.evaluation.mteb \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd
