<h1 align="center">Vis-IR: Unifying Search With Visualized Information Retrieval</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2502.11431">
        <img alt="Build" src="http://img.shields.io/badge/arXiv-2502.11431-B31B1B.svg">
    </a>
    <a href="https://github.com/VectorSpaceLab/Vis-IR">
        <img alt="Build" src="https://img.shields.io/badge/Github-Code-blue">
    </a>
    <a href="https://huggingface.co/datasets/marsh123/VIRA/">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Datasets-VIRA-yellow">
    </a>  
    <a href="https://huggingface.co/datasets/marsh123/MVRB">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Datasets-MVRB-yellow">
    </a>  
    <!-- <a href="">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-UniSE CLIP-yellow">
    </a>  -->
    <a href="https://huggingface.co/marsh123/UniSE">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Model-UniSE MLLM-yellow">
    </a> 
     
</p>
<h4 align="center">
    <p>
        <a href=#news>News</a> |
        <a href=#release-plan>Release Plan</a> |
        <a href=#overview>Overview</a> |
        <a href="#license">License</a> |
        <a href="#citation">Citation</a>
    <p>
</h4>

## News

```2025-04-06``` 🚀🚀 MVRB Dataset are released on Huggingface: [MVRB](https://huggingface.co/datasets/marsh123/MVRB)

```2025-04-02``` 🚀🚀 VIRA Dataset are released on Huggingface: [VIRA](https://huggingface.co/datasets/marsh123/VIRA/)

```2025-04-01``` 🚀🚀 UniSE models are released on Huggingface: [UniSE-MLMM](https://huggingface.co/marsh123/UniSE-MLLM/)

```2025-02-17``` 🎉🎉 Release our paper: [Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval](https://arxiv.org/abs/2502.11431).

## Release Plan
- [x] Paper
- [x] UniSE models
- [x] VIRA Dataset
- [x] MVRB benchmark
- [ ] Evaluation code
- [ ] Fine-tuning code

## Overview

In this work, we formally define an emerging IR paradigm called Visualized Information Retrieval, or **VisIR**, where multimodal information, such as texts, images, tables and charts, is jointly represented by a unified visual format called **Screenshots**, for various retrieval applications. We further make three key contributions for VisIR. First, we create **VIRA** (Vis-IR Aggregation), a large-scale dataset comprising a vast collection of screenshots from diverse sources, carefully curated into captioned and questionanswer formats. Second, we develop **UniSE** (Universal Screenshot Embeddings), a family of retrieval models that enable screenshots to query or be queried across arbitrary data modalities. Finally, we construct **MVRB** (Massive Visualized IR Benchmark), a comprehensive benchmark covering a variety of task forms and application scenarios. Through extensive evaluations on MVRB, we highlight the deficiency from existing multimodal retrievers and the substantial improvements made by UniSE.

## Model Usage

> Our code works well on transformers==4.45.2, and we recommend using this version.

### 1. UniSE-MLLM Models

```python
import torch
from transformers import AutoModel

MODEL_NAME = "marsh123/UniSE-MLLM"
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
                                        # You must set trust_remote_code=True
model.set_processor(MODEL_NAME)

with torch.no_grad():
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    query_inputs = model.data_process(
        images=["./assets/query_1.png", "./assets/query_2.png"],    
        text=["After a 17% drop, what is Nvidia's closing stock price?",
              "I would like to see a detailed and intuitive performance comparison between the two models."],
        q_or_c="query",
        task_instruction="Represent the given image with the given query."
    )
    candidate_inputs = model.data_process(
        images=["./assets/positive_1.jpeg", "./assets/neg_1.jpeg",
                "./assets/positive_2.jpeg", "./assets/neg_2.jpeg"],
        q_or_c="candidate"
    )
    query_embeddings = model(**query_inputs)
    candidate_embeddings = model(**candidate_inputs)
    scores = torch.matmul(query_embeddings, candidate_embeddings.T)
    print(scores)
```

## Performance on MVRB

MVRB is a comprehensive benchmark designed for the retrieval task centered on screenshots. It includes four meta tasks: Screenshot Retrieval (SR), Composed Screenshot Retrieval (CSR), Screenshot QA (SQA), and Open-Vocabulary Classification (OVC). We evaluate three main types of retrievers on MVRB: OCR+Text Retrievers, General Multimodal Retrievers, and Screenshot Document Retrievers. Our proposed UniSE-MLLM achieves state-of-the-art (SOTA) performance on this benchmark.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/66164f6245336ca774679611/igMgX-BvQ55Dyxuw26sgs.png)



## License
Vis-IR is licensed under the [MIT License](LICENSE). 


## Citation
If you find this repository useful, please consider giving a star ⭐ and citation

```
@article{liu2025any,
  title={Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval},
  author={Liu, Ze and Liang, Zhengyang and Zhou, Junjie and Liu, Zheng and Lian, Defu},
  journal={arXiv preprint arXiv:2502.11431},
  year={2025}
}
```
