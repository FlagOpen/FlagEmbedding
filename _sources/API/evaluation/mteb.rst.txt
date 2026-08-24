MTEB
====

`MTEB <https://github.com/embeddings-benchmark/mteb>`_ (The Massive Text Embedding Benchmark) is a large-scale evaluation framework designed to assess the performance of text embedding models across a wide variety of NLP tasks. 
Introduced to standardize and improve the evaluation of text embeddings, MTEB is crucial for assessing how well these models generalize across various real-world applications. 
It contains a wide range of datasets in eight main NLP tasks and different languages, and provides an easy pipeline for evaluation.
It also holds the well known MTEB `leaderboard <https://huggingface.co/spaces/mteb/leaderboard>`_, which contains a ranking of the latest first-class embedding models.

You can evaluate model's performance on the whole MTEB benchmark by running our provided shell script:

.. code:: bash

    chmod +x /examples/evaluation/mteb/eval_mteb.sh
    ./examples/evaluation/mteb/eval_mteb.sh

Or by running:

.. code:: bash

    python -m FlagEmbedding.evaluation.mteb \
    --eval_name mteb \
    --output_dir ./mteb/search_results \
    --languages eng \
    --tasks NFCorpus BiorxivClusteringS2S SciDocsRR \
    --eval_output_path ./mteb/mteb_eval_results.json \
    --embedder_name_or_path BAAI/bge-large-en-v1.5 \
    --devices cuda:7 \
    --cache_dir /root/.cache/huggingface/hub

change the embedder, devices and cache directory to your preference.

.. toctree::
   :hidden:

   mteb/arguments
   mteb/searcher
   mteb/runner