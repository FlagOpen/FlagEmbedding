cd ./code-rag-bench/retrieval/

output_dir='result'

for dataset_name in "humaneval" "mbpp" "repoeval" "ds1000_all_completion" "odex_en" "swe-bench-lite"
do
echo "dataset_name: ${dataset_name}"
python main.py \
    --embedder_name_or_path BAAI/bge-code-v1 \
    --embedder_model_class decoder-only-base \
    --query_instruction_format_for_retrieval '<instruct>{}\n<query>{}' \
    --embedder_query_max_length 2048 \
    --embedder_passage_max_length 2048 \
    --trust_remote_code True \
    --pooling_method last_token \
    --embedder_batch_size 64 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir ./cache \
    --dataset $dataset_name \
    --output_file ../../${output_dir}/${dataset_name}_output.json \
    --results_file ../../${output_dir}/${dataset_name}_results.json
done