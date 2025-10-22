MKQA
====

`MKQA <https://github.com/apple/ml-mkqa>`_ is an open-domain question answering evaluation set comprising 10k question-answer pairs aligned across 26 typologically diverse languages.
The queries are sampled from the [Google Natural Questions Dataset](https://github.com/google-research-datasets/natural-questions). 

Each example in the dataset has the following structure:

.. code:: bash

    {
        'example_id': 563260143484355911,
        'queries': {
            'en': "who sings i hear you knocking but you can't come in",
            'ru': "кто поет i hear you knocking but you can't come in",
            'ja': '「 I hear you knocking」は誰が歌っていますか',
            'zh_cn': "《i hear you knocking but you can't come in》是谁演唱的",
            ...
        },
        'query': "who sings i hear you knocking but you can't come in",
        'answers': {
            'en': [{
                'type': 'entity',
                'entity': 'Q545186',
                'text': 'Dave Edmunds',
                'aliases': [],
            }],
            'ru': [{
                'type': 'entity',
                'entity': 'Q545186',
                'text': 'Эдмундс, Дэйв',
                'aliases': ['Эдмундс', 'Дэйв Эдмундс', 'Эдмундс Дэйв', 'Dave Edmunds'],
            }],
            'ja': [{
                'type': 'entity',
                'entity': 'Q545186',
                'text': 'デイヴ・エドモンズ',
                'aliases': ['デーブ・エドモンズ', 'デイブ・エドモンズ'],
            }],
            'zh_cn': [{
                'type': 'entity', 
                'text': '戴维·埃德蒙兹 ', 
                'entity': 'Q545186',
            }],
            ...
        },
    }


You can evaluate model's performance on MKQA simply by running our provided shell script:

.. code:: bash

    chmod +x /examples/evaluation/mkqa/eval_mkqa.sh
    ./examples/evaluation/mkqa/eval_mkqa.sh

Or by running:

.. code:: bash

    python -m FlagEmbedding.evaluation.mkqa \
    --eval_name mkqa \
    --dataset_dir ./mkqa/data \
    --dataset_names en zh_cn \
    --splits test \
    --corpus_embd_save_dir ./mkqa/corpus_embd \
    --output_dir ./mkqa/search_results \
    --search_top_k 1000 \
    --rerank_top_k 100 \
    --cache_path /root/.cache/huggingface/hub \
    --overwrite False \
    --k_values 20 \
    --eval_output_method markdown \
    --eval_output_path ./mkqa/mkqa_eval_results.md \
    --eval_metrics qa_recall_at_20 \
    --embedder_name_or_path BAAI/bge-m3 \
    --reranker_name_or_path BAAI/bge-reranker-v2-m3 \
    --devices cuda:0 cuda:1 \
    --cache_dir /root/.cache/huggingface/hub \
    --reranker_max_length 1024

change the embedder, reranker, devices and cache directory to your preference.

.. toctree::
   :hidden:

   mkqa/data_loader
   mkqa/evaluator
   mkqa/runner