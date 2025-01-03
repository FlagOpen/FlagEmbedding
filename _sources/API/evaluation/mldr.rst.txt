MLDR
====

`MLDR <https://huggingface.co/datasets/Shitao/MLDR>`_ is a Multilingual Long-Document Retrieval dataset built on Wikipeida, Wudao and mC4, covering 13 typologically diverse languages. 
Specifically, we sample lengthy articles from Wikipedia, Wudao and mC4 datasets and randomly choose paragraphs from them. 
Then we use GPT-3.5 to generate questions based on these paragraphs. 
The generated question and the sampled article constitute a new text pair to the dataset.

An example of ``train`` set looks like:

.. code:: bash

    {
        'query_id': 'q-zh-<...>', 
        'query': '...', 
        'positive_passages': [
            {
                'docid': 'doc-zh-<...>',
                'text': '...'
            }
        ],
        'negative_passages': [
            {
                'docid': 'doc-zh-<...>',
                'text': '...'
            },
            ...
        ]
    }

An example of ``dev`` and ``test`` set looks like:

.. code:: bash

    {
        'query_id': 'q-zh-<...>', 
        'query': '...', 
        'positive_passages': [
            {
                'docid': 'doc-zh-<...>',
                'text': '...'
            }
        ],
        'negative_passages': []
    }

An example of ``corpus`` looks like:

.. code:: bash

    {
        'docid': 'doc-zh-<...>', 
        'text': '...'
    }

You can evaluate model's performance on MLDR simply by running our provided shell script:

.. code:: bash

    chmod +x /examples/evaluation/mldr/eval_mldr.sh
    ./examples/evaluation/mldr/eval_mldr.sh

Or by running:

.. code:: bash

    python -m FlagEmbedding.evaluation.mldr \
    --eval_name mldr \
    --dataset_dir ./mldr/data \
    --dataset_names hi \
    --splits test \
    --corpus_embd_save_dir ./mldr/corpus_embd \
    --output_dir ./mldr/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path /root/.cache/huggingface/hub \
    --overwrite False \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./mldr/mldr_eval_results.md \
    --eval_metrics ndcg_at_10 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir /root/.cache/huggingface/hub \
    --embedder_passage_max_length 8192 \
    --reranker_max_length 8192

change the args of embedder, reranker, devices and cache directory to your preference.

.. toctree::
   :hidden:

   mldr/data_loader
   mldr/runner