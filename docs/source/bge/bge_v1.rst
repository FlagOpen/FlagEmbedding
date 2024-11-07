BGE-v1
======

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

Then to enhance its retrieval ability without instruction and alleviate the issue of the similarity distribution, :code:`bge-*-1.5` models 
were released in Sep 2023. They are still the most popular embedding models that balanced well between embedding quality and model sizes.

+-----------------------------------------------------------------------------+-----------+------------+--------------+--------------+
|                                  Model                                      |  Language | Parameters |  Model Size  |  Description |
+=============================================================================+===========+============+==============+==============+
| `BAAI/bge-large-en-v1.5 <https://huggingface.co/BAAI/bge-large-en-v1.5>`_   |  English  |    335M    |    1.34 GB   | version 1.5  |
+-----------------------------------------------------------------------------+-----------+------------+--------------+ with more    +
| `BAAI/bge-base-en-v1.5 <https://huggingface.co/BAAI/bge-base-en-v1.5>`_     |  English  |    109M    |    438 MB    | reasonable   |
+-----------------------------------------------------------------------------+-----------+------------+--------------+ similarity   +
| `BAAI/bge-small-en-v1.5 <https://huggingface.co/BAAI/bge-small-en-v1.5>`_   |  English  |    33.4M   |    133 MB    | distribution |
+-----------------------------------------------------------------------------+-----------+------------+--------------+              +
| `BAAI/bge-large-zh-v1.5 <https://huggingface.co/BAAI/bge-large-zh-v1.5>`_   |  Chinese  |    326M    |    1.3 GB    |              |
+-----------------------------------------------------------------------------+-----------+------------+--------------+              +
| `BAAI/bge-base-zh-v1.5 <https://huggingface.co/BAAI/bge-base-zh-v1.5>`_     |  Chinese  |    102M    |    409 MB    |              |
+-----------------------------------------------------------------------------+-----------+------------+--------------+              +
| `BAAI/bge-small-zh-v1.5 <https://huggingface.co/BAAI/bge-small-zh-v1.5>`_   |  Chinese  |    24M     |    95.8 MB   |              |
+-----------------------------------------------------------------------------+-----------+------------+--------------+--------------+



