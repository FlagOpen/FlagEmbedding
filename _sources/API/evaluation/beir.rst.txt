BEIR
====

`BEIR <https://github.com/beir-cellar/beir>`_ (Benchmarking-IR) is a heterogeneous evaluation benchmark for information retrieval. 
It is designed for evaluating the performance of NLP-based retrieval models and widely used by research of modern embedding models.

You can evaluate model's performance on the BEIR benchmark by running our provided shell script:

.. code:: bash

    chmod +x /examples/evaluation/beir/eval_beir.sh
    ./examples/evaluation/beir/eval_beir.sh

Or by running:

.. code:: bash

    python -m FlagEmbedding.evaluation.beir \
    --eval_name beir \
    --dataset_dir ./beir/data \
    --dataset_names fiqa arguana cqadupstack \
    --splits test dev \
    --corpus_embd_save_dir ./beir/corpus_embd \
    --output_dir ./beir/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path /root/.cache/huggingface/hub \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./beir/beir_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --ignore_identical_ids True \
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --reranker_max_length 1024 \

change the embedder, devices and cache directory to your preference.

.. toctree::
   :hidden:

   beir/arguments
   beir/data_loader
   beir/evaluator
   beir/runner