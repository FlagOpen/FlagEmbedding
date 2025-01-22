======
BGE-M3
======

BGE-M3 is a compound and powerful embedding model distinguished for its versatility in:
- **Multi-Functionality**: It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval.
- **Multi-Linguality**: It can support more than 100 working languages.
- **Multi-Granularity**: It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens.

+-------------------------------------------------------------------+-----------------+------------+--------------+-----------------------------------------------------------------------+
|                                  Model                            |    Language     | Parameters |  Model Size  |                              Description                              |
+===================================================================+=================+============+==============+=======================================================================+
| `BAAI/bge-m3 <https://huggingface.co/BAAI/bge-m3>`_               |  Multi-Lingual  |    569M    |    2.27 GB   | Multi-Functionality, Multi-Linguality, and Multi-Granularity          |
+-------------------------------------------------------------------+-----------------+------------+--------------+-----------------------------------------------------------------------+

Multi-Linguality
================

BGE-M3 was trained on multiple datasets covering up to 170+ different languages. 
While the amount of training data on languages are highly unbalanced, the actual model performance on different languages will have difference.

For more information of datasets and evaluation results, please check out our `paper <https://arxiv.org/pdf/2402.03216s>`_ for details.

Multi-Granularity
=================

We extend the max position to 8192, enabling the embedding of larger corpus. 
Proposing a simple but effective method: MCLS (Multiple CLS) to enhance the model's ability on long text without additional fine-tuning.

Multi-Functionality
===================

.. code:: python

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('BAAI/bge-m3')
    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
                   "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

Dense Retrieval
---------------

Similar to BGE v1 or v1.5 models, BGE-M3 use the normalized hidden state of the special token [CLS] as the dense embedding:

.. math:: e_q = norm(H_q[0])

Next, to compute the relevance score between the query and passage:

.. math:: s_{dense}=f_{sim}(e_p, e_q)

where :math:`e_p, e_q` are the embedding vectors of passage and query, respectively.

:math:`f_{sim}` is the score function (such as inner product and L2 distance) for comupting two embeddings' similarity.

Sparse Retrieval
----------------

BGE-M3 generates sparce embeddings by adding a linear layer and a ReLU activation function following the hidden states:

.. math:: w_{qt} = \text{Relu}(W_{lex}^T H_q [i])

where :math:`W_{lex}` representes the weights of linear layer and :math:`H_q[i]` is the encoder's output of the :math:`i^{th}` token.

Based on the tokens' weights of query and passage, the relevance score between them is computed by the joint importance of the co-existed terms within the query and passage:

.. math:: s_{lex} = \sum_{t\in q\cap p}(w_{qt} * w_{pt})

where :math:`w_{qt}, w_{pt}` are the importance weights of each co-existed term :math:`t` in query and passage, respectively.

Multi-Vector
------------

The multi-vector method utilizes the entire output embeddings for the representation of query :math:`E_q` and passage :math:`E_p`.

.. math:: 

    E_q = norm(W_{mul}^T H_q)

    E_p = norm(W_{mul}^T H_p)

where :math:`W_{mul}` is the learnable projection matrix.

Following ColBert, BGE-M3 use late-interaction to compute the fine-grained relevance score:

.. math:: s_{mul}=\frac{1}{N}\sum_{i=1}^N\max_{j=1}^M E_q[i]\cdot E_p^T[j]

where :math:`E_q, E_p` are the entire output embeddings of query and passage, respectively.

This is a summation of average of maximum similarity of each :math:`v\in E_q` with vectors in :math:`E_p`.

Hybrid Ranking
--------------

BGE-M3's multi-functionality gives the possibility of hybrid ranking to improve retrieval. 
Firstly, due to the heavy cost of multi-vector method, we can retrieve the candidate results by either of the dense or sparse method. 
Then, to get the final result, we can rerank the candidates based on the integrated relevance score:

.. math:: s_{rank} = w_1\cdot s_{dense}+w_2\cdot s_{lex} + w_3\cdot s_{mul}

where the values chosen for :math:`w_1`, :math:`w_2` and :math:`w_3` varies depending on the downstream scenario.


Usage
=====

.. code:: python

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('BAAI/bge-m3')

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]

    output = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    dense, sparse, multiv = output['dense_vecs'], output['lexical_weights'], output['colbert_vecs']

Useful Links:

`API <../API/inference/embedder/encoder_only/M3Embedder>`_
`Tutorial <https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/1_Embedding/1.2.4_BGE-M3.ipynb>`_
`Example <https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder/encoder_only>`_