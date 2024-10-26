python __main__.py \
--dataset_dir /share/chaofan/code/FlagEmbedding_update/data/BEIR \
--embedder_name_or_path BAAI/bge-large-en-v1.5 \
--reranker_name_or_path BAAI/bge-reranker-v2-m3 \
--query_instruction_for_retrieval "Represent this sentence for searching relevant passages: " \
--use_fp16 True \
--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--cache_dir /share/shared_models \
--corpus_embd_save_dir /share/chaofan/code/FlagEmbedding_update/data/BEIR_passage_embds \
--reranker_max_length 1024 \
--dataset_names trec-covid webis-touche2020 \
--use_special_instructions False

