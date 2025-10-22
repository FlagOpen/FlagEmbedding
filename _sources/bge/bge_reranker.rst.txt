BGE-Reranker
============

Different from embedding model, reranker, or cross-encoder uses question and document as input and directly output similarity instead of embedding. 
To balance the accuracy and time cost, cross-encoder is widely used to re-rank top-k documents retrieved by other simple models. 
For examples, use a bge embedding model to first retrieve top 100 relevant documents, and then use bge reranker to re-rank the top 100 document to get the final top-3 results.

The first series of BGE-Reranker contains two models, large and base.

+-------------------------------------------------------------------------------+-----------------------+------------+--------------+-----------------------------------------------------------------------+
|                                    Model                                      |        Language       | Parameters |  Model Size  |                              Description                              |
+===============================================================================+=======================+============+==============+=======================================================================+
| `BAAI/bge-reranker-large <https://huggingface.co/BAAI/bge-reranker-large>`_   |   English & Chinese   |    560M    |    2.24 GB   | Larger reranker model, easy to deploy with better inference           |
+-------------------------------------------------------------------------------+-----------------------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-reranker-base <https://huggingface.co/BAAI/bge-reranker-base>`_     |   English & Chinese   |    278M    |    1.11 GB   | Lightweight reranker model, easy to deploy with fast inference        |
+-------------------------------------------------------------------------------+-----------------------+------------+--------------+-----------------------------------------------------------------------+

bge-reranker-large and bge-reranker-base used `XLM-RoBERTa-Large <https://huggingface.co/FacebookAI/xlm-roberta-large>`_ and `XLM-RoBERTa-Base <https://huggingface.co/FacebookAI/xlm-roberta-base>`_ respectively as the base model. 
They were trained on high quality English and Chinese data, and acheived State-of-The-Art performance in the level of same size models at the time released.

Usage
-----
    

.. code:: python

    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(
        'BAAI/bge-reranker-base', 
        query_max_length=256,
        use_fp16=True,
        devices=['cuda:1'],
    )

    score = reranker.compute_score(['I am happy to help', 'Assisting you is my pleasure'])
    print(score)