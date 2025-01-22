:github_url: https://github.com/AI4Finance-Foundation/FinRL

Installation
============

Using pip:
----------

If you do not need to finetune the models, you can install the package without the finetune dependency:

.. code:: bash

    pip install -U FlagEmbedding

If you want to finetune the models, you can install the package with the finetune dependency:

.. code:: bash

    pip install -U FlagEmbedding[finetune]


Install from sources:
---------------------

Clone the repository and install

.. code:: bash

    git clone https://github.com/FlagOpen/FlagEmbedding.git
    cd FlagEmbedding
    # If you do not need to finetune the models, you can install the package without the finetune dependency:
    pip install  .
    # If you want to finetune the models, install the package with the finetune dependency:
    pip install  .[finetune]

For development in editable mode:

.. code:: bash

    # If you do not need to finetune the models, you can install the package without the finetune dependency:
    pip install -e .
    # If you want to finetune the models, install the package with the finetune dependency:
    pip install -e .[finetune]

PyTorch-CUDA
------------

If you want to use CUDA GPUs during inference and finetuning, please install appropriate version of `PyTorch <https://pytorch.org/get-started/locally/>`_ with CUDA support.