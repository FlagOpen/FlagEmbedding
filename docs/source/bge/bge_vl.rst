BGE-VL
======

BGE-VL is a series of multimodel retrieval models training on `MegaPairs <https://github.com/VectorSpaceLab/MegaPairs>`_ 

BGE-VL contains light weight CLIP based models as well as more powerful LLAVA-NeXT based MLLM models:

+----------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
|                                  Model                               |  Language | Parameters |  Model Size  |                              Description                              |
+======================================================================+===========+============+==============+=======================================================================+
| `BAAI/bge-vl-base <https://huggingface.co/BAAI/BGE-VL-base>`_        |  English  |    150M    |    299 MB    |        Light weight multimodel embedder among image and text          |
+----------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-vl-large <https://huggingface.co/BAAI/BGE-VL-large>`_      |  English  |    428M    |    855 MB    |         Large scale multimodel embedder among image and text          |
+----------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-vl-MLLM-S1 <https://huggingface.co/BAAI/BGE-VL-MLLM-S1>`_  |  English  |    7.57B   |   15.14 GB   |   SOTA in composed image retrieval, trained on MegaPairs dataset      |
+----------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/bge-vl-MLLM-S2 <https://huggingface.co/BAAI/BGE-VL-MLLM-S2>`_  |  English  |    7.57B   |   15.14 GB   |   Finetune BGE-VL-MLLM-S1 with one epoch on MMEB training set         |
+----------------------------------------------------------------------+-----------+------------+--------------+-----------------------------------------------------------------------+
| `BAAI/BGE-VL-v1.5-zs <https://huggingface.co/BAAI/BGE-VL-v1.5-zs>`_    |  English  |   7.57B   |   15.14 GB   |    Better multi-modal retrieval model with performs well in all kinds of tasks    |
| `BAAI/BGE-VL-v1.5-mmeb <https://huggingface.co/BAAI/BGE-VL-v1.5-mmeb>`_    |  English  |   7.57B   |   15.14 GB   |    Better multi-modal retrieval model, additionally fine-tuned on MMEB training set    |


BGE-VL-CLIP
-----------

The base and large model are trained based on CLIP-vit-base-patch16 and CLIP-vit-large-patch14. 
For composed image-text data, the model directly use score-fusion to sum up the outputs of visual encoder and text encoder and get the final embedding.

.. tip::

    Our code works well on transformers==4.45.2, and we recommend using this version.

You can easily use BGE-VL-CLIP models based on transformers:

.. code:: python

    import torch
    from transformers import AutoModel

    MODEL_NAME = "BAAI/BGE-VL-base" # or "BAAI/BGE-VL-large"
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True) # You must set trust_remote_code=True
    model.set_processor(MODEL_NAME)
    model.eval()

    with torch.no_grad():
        query = model.encode(
            images = "./assets/cir_query.png", 
            text = "Make the background dark, as if the camera has taken the photo at night"
        )
        candidates = model.encode(
            images = ["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"]
        )
        
        scores = query @ candidates.T
    print(scores)


BGE-VL-MLLM
-----------

The multimodal large language models (MLLMs) incorporate a visual encoder, typically based on a vision transformer, into a large language model (LLM). 
This integration allows image tokens to be directly processed by the LLM. 
Consequently, MLLMs can effectively handle diverse multimodal inputs by converting any type of input into a sequence of tokens.

BGE-VL-MLLM builds upon the LLaVA1.6. In both training and inference stages, MMRet uses task-specific instructions for query inputs to improve generalization, aligning
with standard practices in LLM-based embedding models. 
A typical multimodal query input is structured as follows:

.. math:: 

    ⟨\text{instruct}⟩{\{task\_ inst\}} \space⟨\text{query}⟩\{q_t\} \{q_i\}\space[\text{EOS}]

where :math:`{task_inst}` represents the task-specific instruction, :math:`{qt}` denotes the input query text, and
:math:`{qi}` is the input query image. 
The normalized last hidden state of the [EOS] token in the MLLM is used as the embedding of any given input sequence.

.. code:: python

    import torch
    from transformers import AutoModel
    from PIL import Image

    MODEL_NAME= "BAAI/BGE-VL-MLLM-S1"
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    model.cuda()

    with torch.no_grad():
        model.set_processor(MODEL_NAME)

        query_inputs = model.data_process(
            text="Make the background dark, as if the camera has taken the photo at night", 
            images="./assets/cir_query.png",
            q_or_c="q",
            task_instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
        )
        candidate_inputs = model.data_process(
            images=["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"],
            q_or_c="c",
        )

        query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
        candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
        
        query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
        candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)

        scores = torch.matmul(query_embs, candi_embs.T)
    print(scores)


BGE-VL-v1.5
-----------

BGE-VL-v1.5 series is the updated version of BGE-VL, bringing better performance on both retrieval and multi-modal understanding. The models were trained on 30M MegaPairs data and extra 10M natural and synthetic data.

`bge-vl-v1.5-zs` is a zero-shot model, only trained on the data mentioned above. `bge-vl-v1.5-mmeb` is the fine-tuned version on MMEB training set.


.. code:: python

    import torch
    from transformers import AutoModel
    from PIL import Image

    MODEL_NAME= "BAAI/BGE-VL-v1.5-mmeb" # "BAAI/BGE-VL-v1.5-zs"

    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    model.cuda()

    with torch.no_grad():
        model.set_processor(MODEL_NAME)

        query_inputs = model.data_process(
            text="Make the background dark, as if the camera has taken the photo at night", 
            images="../../imgs/cir_query.png",
            q_or_c="q",
            task_instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
        )

        candidate_inputs = model.data_process(
            images=["../../imgs/cir_candi_1.png", "../../imgs/cir_candi_2.png"],
            q_or_c="c",
        )

        query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
        candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
        
        query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
        candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)

        scores = torch.matmul(query_embs, candi_embs.T)
    print(scores)



For more details, check out the repo of `MegaPairs <https://github.com/VectorSpaceLab/MegaPairs>`_