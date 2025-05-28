output_dir=result

python main.py \
    --output_dir ${output_dir} \
    --use_special_instructions True \
    --embedder_name_or_path BAAI/bge-code-v1 \
    --embedder_model_class decoder-only-base \
    --query_instruction_format_for_retrieval '<instruct>{}\n<query>{}' \
    --embedder_query_max_length 2048 \
    --embedder_passage_max_length 2048 \
    --trust_remote_code True \
    --pooling_method last_token \
    --embedder_batch_size 64 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --tasks apps codetrans-contest codetrans-dl cosqa synthetic-text2sql stackoverflow-qa codefeedback-mt codefeedback-st CodeSearchNet-ccr-go CodeSearchNet-ccr-java CodeSearchNet-ccr-javascript CodeSearchNet-ccr-php CodeSearchNet-ccr-python CodeSearchNet-ccr-ruby CodeSearchNet-go CodeSearchNet-java CodeSearchNet-javascript CodeSearchNet-php CodeSearchNet-python CodeSearchNet-ruby \
    --cache_dir ./cache