BGE v1 & v1.5
=============

BGE v1 and v1.5 are series of encoder only models base on BERT. They achieved best performance among the models of the same size at the time of release.

BGE
---

The first group of BGE models was released in Aug 2023. The :code:`bge-large-en` and :code:`bge-large-zh` ranked 1st on MTEB and 
C-MTEB benchmarks at the time released.

+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
|                                  Model                            |  Language | Parameters |  Model Size  |                              Description                              |
+===================================================================+===========+============+==============+=======================================================================+
| `BAAI/bge-large-en <https://huggingface.co/BAAI/bge-large-en>`_   |  English  |    335M    |    1.34 GB   | Embedding Model which map text into vector                            |
+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-base-en <https://huggingface.co/BAAI/bge-base-en>`_     |  English  |    109M    |    438 MB    | a base-scale model but with similar ability to `BAAI/bge-large-en`    |
+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-small-en <https://huggingface.co/BAAI/bge-small-en>`_   |  English  |    33.4M   |    133 MB    | a small-scale model but with competitive performance                  |
+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-large-zh <https://huggingface.co/BAAI/bge-large-zh>`_   |  Chinese  |    326M    |    1.3 GB    | Embedding Model which map text into vector                            |
+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-base-zh <https://huggingface.co/BAAI/bge-base-zh>`_     |  Chinese  |    102M    |    409 MB    | a base-scale model but with similar ability to `BAAI/bge-large-zh`    |
+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-small-zh <https://huggingface.co/BAAI/bge-small-zh>`_   |  Chinese  |    24M     |    95.8 MB   | a small-scale model but with competitive performance                  |
+-------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+

BGE-v1.5
--------

Then to enhance its retrieval ability without instruction and alleviate the issue of the similarity distribution, :code:`bge-*-v1.5` models 
were released in Sep 2023. They are still the most popular embedding models that balanced well between embedding quality and model sizes.

+-----------------------------------------------------------------------------+-----------+------------+--------------+--------------+
|                                  Model                                      |  Language | Parameters |  Model Size  |  Description |
+=============================================================================+===========+============+==============+==============+
| `BAAI/bge-large-en-v1.5 <https://huggingface.co/BAAI/bge-large-en-v1.5>`_   |  English  |    335M    |    1.34 GB   | version 1.5  |
+-----------------------------------------------------------------------------+-----------+------------+--------------+ with more    +
| `BAAI/bge-base-en-v1.5 <https://huggingface.co/BAAI/bge-base-en-v1.5>`_     |  English  |    109M    |    438 MB    | reasonable   |
+-----------------------------------------------------------------------------+-----------+------------+--------------+ similarity   +
| `BAAI/bge-small-en-v1.5 <https://huggingface.co/BAAI/bge-small-en-v1.5>`_   |  English  |    33.4M   |    133 MB    | distribution |
+-----------------------------------------------------------------------------+-----------+------------+--------------+ and better   +
| `BAAI/bge-large-zh-v1.5 <https://huggingface.co/BAAI/bge-large-zh-v1.5>`_   |  Chinese  |    326M    |    1.3 GB    | performance  |
+-----------------------------------------------------------------------------+-----------+------------+--------------+              +
| `BAAI/bge-base-zh-v1.5 <https://huggingface.co/BAAI/bge-base-zh-v1.5>`_     |  Chinese  |    102M    |    409 MB    |              |
+-----------------------------------------------------------------------------+-----------+------------+--------------+              +
| `BAAI/bge-small-zh-v1.5 <https://huggingface.co/BAAI/bge-small-zh-v1.5>`_   |  Chinese  |    24M     |    95.8 MB   |              |
+-----------------------------------------------------------------------------+-----------+------------+--------------+--------------+


Usage
-----

To use BGE v1 or v1.5 model for inference, load model through

.. code:: python

    from FlagEmbedding import FlagModel

    model = FlagModel('BAAI/bge-base-en-v1.5')

    sentences = ["Hello world", "I am inevitable"]
    embeddings = model.encode(sentences)

.. tip::

    For simple tasks that only encode a few sentences like above, it's faster to use CPU or a single GPU instead of multi-GPUs

To use CPU:

.. code:: python

    # make no GPU visible
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # or claim the devices during initialize the model
    model = FlagModel('BAAI/bge-base-en-v1.5', devices='cpu')

To use a single GPU:

.. code:: python

    # select one sigle card to be visible
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # or claim the devices during initialize the model
    model = FlagModel('BAAI/bge-base-en-v1.5', devices=0)

|

Useful Links:

`API <../API/inference/embedder/encoder_only/BaseEmbedder>`_

`Tutorial <https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/1_Embedding/1.2.3_BGE_v1%261.5.ipynb>`_

`Example <https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder/encoder_only>`_