.. FlagEmbedding documentation master file, created by
   sphinx-quickstart on Sat Oct 12 13:27:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BAAI General Embedding
======================

|
|

.. image:: _static/img/bge_logo.jpg
   :target: https://github.com/FlagOpen/FlagEmbedding
   :width: 500
   :align: center

|
|

Welcome to BGE documentation!

We aim for building one-stop retrieval toolkit for search and RAG.

Besides the resources we provide here in this documentation, please visit our `GitHub repo <https://github.com/FlagOpen/FlagEmbedding>`_ for more contents including:

- Want to get familiar with BGE quickly? There are hands-on `examples <https://github.com/FlagOpen/FlagEmbedding/tree/068e86f58eccc3107aacb119920de8dba9caa913/examples>`_ to run for embedder and reranker's inference, evaluation, and finetuning.
- Unfamiliar with some area, keywords or techniques of retrieval and RAG? We provide `tutorials <https://github.com/FlagOpen/FlagEmbedding/tree/068e86f58eccc3107aacb119920de8dba9caa913/Tutorials>`_ to teach you basic knowledge and coding tips.
- Interested in research topics that expanding from BGE and retrieval? Our `research <https://github.com/FlagOpen/FlagEmbedding/tree/068e86f58eccc3107aacb119920de8dba9caa913/research>`_ folder contains many exciting topics for you to explore.

BGE is developed by Beijing Academy of Artificial Intelligence (BAAI).

|

.. image:: _static/img/BAAI_logo.png
   :target: https://github.com/FlagOpen/FlagEmbedding
   :width: 300
   :align: center


.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   Introduction/installation
   Introduction/quick_start

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: API

   API/abc
   API/inference
   API/evaluation
   API/finetune

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorial/1_Embedding
   tutorial/2_Metrics
   tutorial/3_Indexing
   tutorial/4_Evaluation
   tutorial/5_Reranking
   tutorial/6_RAG