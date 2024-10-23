python __main__.py \
--dataset_dir /share/chaofan/code/FlagEmbedding_update/data \
--embedder_name_or_path BAAI/bge-large-en-v1.5 \
--use_fp16 True \
--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
--cache_dir /share/shared_models \
--corpus_embd_save_dir /share/chaofan/code/FlagEmbedding_update/data/passage_embds \
--reranker_name_or_path BAAI/bge-reranker-large \
--reranker_max_length 512 \
--splits "dev dl19 dl20"