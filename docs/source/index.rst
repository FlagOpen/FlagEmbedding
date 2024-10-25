.. FlagEmbedding documentation master file, created by
   sphinx-quickstart on Sat Oct 12 13:27:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FlagEmbedding
=============

|
|

.. image:: _static/img/BAAI_logo.png
   :target: https://github.com/FlagOpen/FlagEmbedding
   :width: 400
   :align: center

|
|

Welcome to FlagEmbedding documentation! 

`FlagEmbedding <https://github.com/FlagOpen/FlagEmbedding>`_ focuses on retrieval-augmented LLMs, 
developed with the support of the Beijing Academy of Artificial Intelligence (BAAI).
We are aiming to enhance text and multi-model retrieval by leveraging advanced embedding techniques. 

- We provide high quality text embedding models and rerankers, with multi-language and multi-model, in `BGE <./bge/introduction.html>`_ series. 
- We construct a benchmark for chinese text embedding `C-MTEB <./cmteb/introduction.html>`_, which has been merged into MTEB.




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
   :maxdepth: 1
   :caption: BGE

   bge/introduction
   bge/bge_v1
   bge/llm_embedder
   bge/bge_m3
   bge/bge_icl
   bge/bge_reranker

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Projects

   projects/C-MTEB
   projects/MLVU
   projects/Visualized_BGE

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