.. FlagEmbedding documentation master file, created by
   sphinx-quickstart on Sat Oct 12 13:27:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:html_theme.sidebar_secondary.remove: True


Welcome to BGE!
===============

.. Welcome to BGE documentation!

.. figure:: _static/img/bge_panda.jpg
   :width: 400
   :align: center

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :octicon:`milestone` Introduction

      New to BGE? Quickly get hands-on information.

      +++

      .. button-ref:: Introduction/index
         :expand:
         :color: primary
         :click-parent:

         To Introduction


   .. grid-item-card:: :octicon:`package` BGE Models

      Get to know BGE embedding models and rerankers.

      +++

      .. button-ref:: bge/index
         :expand:
         :color: primary
         :click-parent:

         To BGE


   .. grid-item-card:: :octicon:`log` Tutorials

      Find useful tutorials to start with if you are looking for guidance

      +++

      .. button-ref:: tutorial/index
         :expand:
         :color: primary
         :click-parent:

         To Tutorials

   .. grid-item-card:: :octicon:`codescan` API

      Check the API of classes and functions in FlagEmbedding.

      +++

      .. button-ref:: API/index
         :expand:
         :color: primary
         :click-parent:

         To APIs

   .. grid-item-card:: :octicon:`question` FAQ

      Take a look of questions people frequently asked.

      +++

      .. button-ref:: FAQ/index
         :expand:
         :color: primary
         :click-parent:

         To FAQ
   
   .. grid-item-card:: :octicon:`people` Community

      Welcome to join BGE community!

      +++

      .. button-ref:: community/index
         :expand:
         :color: primary
         :click-parent:

         To Community

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

   Introduction/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: BGE

   bge/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorial/index

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: API

   API/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: FAQ

   FAQ/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Community

   community/index