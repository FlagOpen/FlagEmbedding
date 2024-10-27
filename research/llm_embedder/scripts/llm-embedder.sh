# the instruction and training config version
version="llm-embedder"
# the output folder
output="llm-embedder"
# the data root where you untar the data
data_root="/data/llm-embedder"

torchrun --nproc_per_node=8 run_dense.py --train_data \
    llm-embedder:chat/msc/train.json \
    llm-embedder:convsearch/qrecc/train.concat.json \
    llm-embedder:lrlm/arxiv/train.json \
    llm-embedder:lrlm/books3/train.json \
    llm-embedder:lrlm/codeparrot/train.json \
    llm-embedder:qa/msmarco/train.json \
    llm-embedder:qa/nq/train.json \
    llm-embedder:tool/toolbench/train.json \
    llm-embedder:tool/toolbench/train.json \
    llm-embedder:icl/icl/train.json \
    --output_dir data/outputs/$output \
    --save_steps 10000 \
    --max_steps 10000 \
    --logging_steps 100 \
    --inbatch_same_dataset epoch \
    --use_train_config \
    --gradient_checkpointing \
    --per_device_train_batch_size 100 \
    --deepspeed data/deepspeed/stage0.json \
    --version $version \
    --learning_rate 5e-6 \
    --data_root $data_root

for model in "checkpoint-10000"
do
    torchrun --nproc_per_node 8 -m evaluation.eval_mmlu --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_msc --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_tool --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/books3/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/arxiv/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/codeparrot/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/pg19/test.json --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_icl --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
    torchrun --nproc_per_node 8 -m evaluation.eval_qrecc --query_encoder data/outputs/$output/$model/encoder --version $version --data_root $data_root
done
