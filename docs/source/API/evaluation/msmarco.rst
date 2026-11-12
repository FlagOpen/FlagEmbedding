MSMARCO
=======

`MS Marco <https://microsoft.github.io/msmarco/>`_ (Microsoft MAchine Reading Comprehension) is a large scale real-world reading comprehension dataset.
It is widely used in information retrieval, question answering, and natural language processing research.


You can evaluate model's performance on MS MARCO simply by running our provided shell script:

.. code:: bash

    chmod +x /examples/evaluation/msmarco/eval_msmarco.sh
    ./examples/evaluation/msmarco/eval_msmarco.sh

Or by running:

.. code:: bash

    python -m FlagEmbedding.evaluation.msmarco \
    --eval_name msmarco \
    --dataset_dir ./msmarco/data \
    --dataset_names passage \
    --splits dev \
    --corpus_embd_save_dir ./msmarco/corpus_embd \
    --output_dir ./msmarco/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path /root/.cache/huggingface/hub \
    --overwrite True \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./msmarco/msmarco_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir /root/.cache/huggingface/hub \
    --reranker_max_length 1024

change the embedder, reranker, devices and cache directory to your preference.

.. toctree::
   :hidden:

   msmarco/data_loader
   msmarco/runner