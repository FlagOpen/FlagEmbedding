Quick Start
===========

First, load one of the BGE embedding model:

.. code:: python

    from FlagEmbedding import FlagAutoModel

    model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5')

.. tip::

    If there's difficulty connecting to Hugging Face, you can use the `HF mirror <https://hf-mirror.com/>`_ instead.

    .. code:: bash

        export HF_ENDPOINT=https://hf-mirror.com

Then, feed some sentences to the model and get their embeddings:

.. code:: python

    sentences_1 = ["I love NLP", "I love machine learning"]
    sentences_2 = ["I love BGE", "I love text retrieval"]
    embeddings_1 = model.encode(sentences_1)
    embeddings_2 = model.encode(sentences_2)

Once we get the embeddings, we can compute similarity by inner product:

.. code:: python

    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
