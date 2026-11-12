BGE-Reranker-v2
===============

+------------------------------------------------------------------------------------------------------------------+-----------------------+-------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                      Model                                                       |        Language       | Parameters  |  Model Size  |                                                                       Description                                                                       |
+==================================================================================================================+=======================+=============+==============+=========================================================================================================================================================+
| `BAAI/bge-reranker-v2-m3 <https://huggingface.co/BAAI/bge-reranker-v2-m3>`_                                      |      Multilingual     |    568M     |    2.27 GB   | Lightweight reranker model, possesses strong multilingual capabilities, easy to deploy, with fast inference.                                            |
+------------------------------------------------------------------------------------------------------------------+-----------------------+-------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| `BAAI/bge-reranker-v2-gemma <https://huggingface.co/BAAI/bge-reranker-v2-gemma>`_                                |      Multilingual     |    2.51B    |     10 GB    | Suitable for multilingual contexts, performs well in both English proficiency and multilingual capabilities.                                            |
+------------------------------------------------------------------------------------------------------------------+-----------------------+-------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| `BAAI/bge-reranker-v2-minicpm-layerwise <https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise>`_        |      Multilingual     |    2.72B    |    10.9 GB   | Suitable for multilingual contexts, allows freedom to select layers for output, facilitating accelerated inference.                                     |
+------------------------------------------------------------------------------------------------------------------+-----------------------+-------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| `BAAI/bge-reranker-v2.5-gemma2-lightweight <https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight>`_  |      Multilingual     |    2.72B    |    10.9 GB   | Suitable for multilingual contexts, allows freedom to select layers, compress ratio and compress layers for output, facilitating accelerated inference. |
+------------------------------------------------------------------------------------------------------------------+-----------------------+-------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+


.. tip::

    You can select the model according your senario and resource:

    - For multilingual, utilize :code:`BAAI/bge-reranker-v2-m3`, :code:`BAAI/bge-reranker-v2-gemma` and :code:`BAAI/bge-reranker-v2.5-gemma2-lightweight`.
    - For Chinese or English, utilize :code:`BAAI/bge-reranker-v2-m3` and :code:`BAAI/bge-reranker-v2-minicpm-layerwise`.
    - For efficiency, utilize :code:`BAAI/bge-reranker-v2-m3` and the low layer of :code:`BAAI/bge-reranker-v2-minicpm-layerwise`.
    - For better performance, recommand :code:`BAAI/bge-reranker-v2-minicpm-layerwise` and :code:`BAAI/bge-reranker-v2-gemma`.

    Make sure always test on your real use case and choose the one with best speed-quality balance!

Usage
-----

**bge-reranker-v2-m3**

Use :code:`bge-reranker-v2-m3` in the same way as bge-reranker-base and bge-reranker-large.

.. code:: python

    from FlagEmbedding import FlagReranker

    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    score = reranker.compute_score(['query', 'passage'])
    # or set "normalize=True" to apply a sigmoid function to the score for 0-1 range
    score = reranker.compute_score(['query', 'passage'], normalize=True)

    print(score)

**bge-reranker-v2-gemma**

Use the :code:`FlagLLMReranker` class for bge-reranker-v2-gemma.

.. code:: python

    from FlagEmbedding import FlagLLMReranker

    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)

    score = reranker.compute_score(['query', 'passage'])
    print(score)

**bge-reranker-v2-minicpm-layerwise**

Use the :code:`LayerWiseFlagLLMReranker` class for bge-reranker-v2-minicpm-layerwise.

.. code:: python

    from FlagEmbedding import LayerWiseFlagLLMReranker

    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)

    # Adjusting 'cutoff_layers' to pick which layers are used for computing the score.
    score = reranker.compute_score(['query', 'passage'], cutoff_layers=[28]) 
    print(score)

**bge-reranker-v2.5-gemma2-lightweight**

Use the :code:`LightWeightFlagLLMReranker` class for bge-reranker-v2.5-gemma2-lightweight.

.. code:: python

    from FlagEmbedding import LightWeightFlagLLMReranker

    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    reranker = LightWeightFlagLLMReranker('BAAI/bge-reranker-v2.5-gemma2-lightweight', use_fp16=True)

    # Adjusting 'cutoff_layers' to pick which layers are used for computing the score.
    score = reranker.compute_score(['query', 'passage'], cutoff_layers=[28], compress_ratio=2, compress_layer=[24, 40])
    print(score)