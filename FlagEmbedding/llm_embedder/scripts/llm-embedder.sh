version="llm-embedder"
output="llm-embedder"

torchrun --nproc_per_node=8 run_dense.py --train_data \
    llm-embedder:chat/msc/train.scored.llama2-7b-chat.json \
    llm-embedder:convsearch/qrecc/train.concat.neg.bm25-from-human.json \
    llm-embedder:lrlm/arxiv/train.128tok.scored.llama2-7b-chat.json \
    llm-embedder:lrlm/books3/train.128tok.scored.llama2-7b-chat.json \
    llm-embedder:lrlm/codeparrot/train.128tok.scored.llama2-7b-chat.json \
    llm-embedder:qa/msmarco/train.wl.json \
    llm-embedder:qa/nq/train.neg.bge.scored.deberta-large.json \
    llm-embedder:tool/toolbench/train.hardneg.json \
    llm-embedder:tool/toolbench/train.hardneg.json \
    llm-embedder:icl/icl/train.scored.llama2-chat.top20.json \
    --output_dir /share/peitian/Code/LlamaRetriever/data/outputs/$output \
    --save_steps 5000 \
    --max_steps 30000 \
    --early_exit_steps 10000 \
    --logging_steps 100 \
    --inbatch_same_dataset epoch \
    --use_train_config \
    --gradient_checkpointing \
    --per_device_train_batch_size 100 \
    --deepspeed /share/peitian/Code/LlamaRetriever/data/deepspeed/stage0.json \
    --version $version \
    --learning_rate 5e-6

for model in "checkpoint-5000" "checkpoint-10000"
do
    torchrun --nproc_per_node 8 -m evaluation.eval_mmlu --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_qa --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_chat --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_tool --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/books3/test.json --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/arxiv/test.json --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/codeparrot/test.json --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_lrlm --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --eval_data llm-embedder:lrlm/pg19/test.json --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_icl --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
    torchrun --nproc_per_node 8 -m evaluation.eval_convsearch --query_encoder /share/peitian/Code/LlamaRetriever/data/outputs/$output/$model/encoder --version $version
done
