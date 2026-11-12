FAQ
===

Below are some commonly asked questions.

.. tip::

    For more questions, search in issues on GitHub or join our community!

.. dropdown:: Having network issue when connecting to Hugging Face?
    :animate: fade-in-slide-down

    Try to set the :code:`HF_ENDPOINT` to `HF mirror <https://hf-mirror.com/>`_ instead.

    .. code:: bash

        export HF_ENDPOINT=https://hf-mirror.com

.. dropdown:: When does the query instruction need to be used?
    :animate: fade-in-slide-down

    For a retrieval task that uses short queries to find long related documents, it is recommended to add instructions for these short queries. 
    The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task. 
    In all cases, the documents/passages do not need to add the instruction.

.. dropdown:: Why it takes quite long to just encode 1 sentence?
    :animate: fade-in-slide-down

    Note that if you have multiple CUDA GPUs, FlagEmbedding will automatically use all of them. 
    Then the time used to start the multi-process will cost way longer than the actual encoding.
    Try to just use CPU or just single GPU for simple tasks.

.. dropdown:: The embedding results are different for CPU and GPU?
    :animate: fade-in-slide-down

    The encode function will use FP16 by default if GPU is available, which leads to different precision. 
    Set :code:`fp16=False` to get full precision.

.. dropdown:: How many languages do the multi-lingual models support?
    :animate: fade-in-slide-down

    The training datasets cover up to 170+ languages. 
    But note that due to the unbalanced distribution of languages, the performances will be different.
    Please further test refer to the real application scenario.

.. dropdown:: How does the different retrieval method works in bge-m3?
    :animate: fade-in-slide-down

    - Dense retrieval: map the text into a single embedding, e.g., `DPR <https://arxiv.org/abs/2004.04906>`_, `BGE-v1.5 <../bge/bge_v1_v1.5>`_
    - Sparse retrieval (lexical matching): a vector of size equal to the vocabulary, with the majority of positions set to zero, calculating a weight only for tokens present in the text. e.g., BM25, `unicoil <https://arxiv.org/pdf/2106.14807>`_, and `splade <https://arxiv.org/abs/2107.05720>`_
    - Multi-vector retrieval: use multiple vectors to represent a text, e.g., `ColBERT <https://arxiv.org/abs/2004.12832>`_.

.. dropdown:: Recommended vector database?
    :animate: fade-in-slide-down

    Generally you can use any vector database (open-sourced, commercial). We use `Faiss <https://github.com/facebookresearch/faiss>`_ by default in our evaluation pipeline and tutorials.

.. dropdown:: No enough VRAM or OOM error during evaluation?
    :animate: fade-in-slide-down

    The default values of :code:`embedder_batch_size` and :code:`reranker_batch_size` are both 3000. Try a smaller value.
