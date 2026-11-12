cd ./code-rag-bench/retrieval/

for dataset_name in "humaneval" "mbpp" "live_code_bench" "ds1000" "odex" "repoeval_repo" "swebench_repo"
do
echo "dataset_name: ${dataset_name}"
PYTHONPATH=./ python create/${dataset_name}.py
done