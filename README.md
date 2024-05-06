<h1 align="center">FlagEmbedding</h1>
<p align="center">
    <a href="https://github.com/FlagOpen/FlagEmbedding">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="Build" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding">
        <img alt="Build" src="https://img.shields.io/badge/FlagEmbedding-1.1-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#news>News</a> |
        <a href="#projects">Projects</a> |
        <a href=#model-list>Model List</a> |
        <a href="#contributor">Contributor</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>


[English](README.md) | [中文](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

FlagEmbedding focuses on retrieval-augmented LLMs, consisting of the following projects currently:

- **Long-Context LLM**: [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon), [LongLLM QLoRA](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)
- **Fine-tuning of LM** : [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)
- **Embedding Model**: [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual), [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3), [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)
- **Reranker Model**: [llm rerankers](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker), [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)
- **Benchmark**: [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)

## News 
- 4/30/2024: Release [Llama-3-8B-Instruct-80K-QLoRA](https://huggingface.co/namespace-Pt/Llama-3-8B-Instruct-80K-QLoRA), extending the context length of Llama-3-8B-Instruct from 8K to 80K via QLoRA training on a few synthesized long-context data. The model achieves remarkable performance on various long-context benchmarks. [Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora) :fire:
- 3/18/2024: Release new [rerankers](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker), built upon powerful M3 and LLM (GEMMA and MiniCPM, not so large actually :smiley:) backbones, supporitng multi-lingual processing and larger inputs, massive improvements of ranking performances on BEIR, C-MTEB/Retrieval, MIRACL, LlamaIndex Evaluation :fire:
- 3/18/2024: Release [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual), equipping BGE with visual capabilities. Visualized-BGE can be utilized to generate embeddings for hybrid image-text data. :fire:
- 1/30/2024: Release **BGE-M3**, a new member to BGE model series! M3 stands for **M**ulti-linguality (100+ languages), **M**ulti-granularities (input length up to 8192), **M**ulti-Functionality (unification of dense, lexical, multi-vec/colbert retrieval). 
It is the first embedding model which supports all three retrieval methods, achieving new SOTA on multi-lingual (MIRACL) and cross-lingual (MKQA) benchmarks.
[Technical Report](https://arxiv.org/pdf/2402.03216.pdf) and [Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3). :fire:
- 1/9/2024: Release [Activation-Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon), an effective, efficient, compatible, and low-cost (training) method to extend the context length of LLM. [Technical Report](https://arxiv.org/abs/2401.03462) 
- 12/24/2023: Release **LLaRA**, a LLaMA-7B based dense retriever, leading to state-of-the-art performances on MS MARCO and BEIR. Model and code will be open-sourced. Please stay tuned. [Technical Report](https://arxiv.org/abs/2312.15503) 
- 11/23/2023: Release [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail), a method to maintain general capabilities during fine-tuning by merging multiple language models. [Technical Report](https://arxiv.org/abs/2311.13534) 
- 10/12/2023: Release [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), a unified embedding model to support diverse retrieval augmentation needs for LLMs. [Technical Report](https://arxiv.org/pdf/2310.07554.pdf)
- 09/15/2023: The [technical report](https://arxiv.org/pdf/2309.07597.pdf) of BGE has been released 
- 09/15/2023: The [massive training data](https://data.baai.ac.cn/details/BAAI-MTP) of BGE has been released 
- 09/12/2023: New models: 
    - **New reranker model**: release cross-encoder models `BAAI/bge-reranker-base` and `BAAI/bge-reranker-large`, which are more powerful than embedding model. We recommend to use/fine-tune them to re-rank top-k documents returned by embedding models. 
    - **update embedding model**: release `bge-*-v1.5` embedding model to alleviate the issue of the similarity distribution, and enhance its retrieval ability without instruction.


<details>
  <summary>More</summary>
<!-- ### More -->

- 09/07/2023: Update [fine-tune code](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md): Add script to mine hard negatives and support adding instruction during fine-tuning. 
- 08/09/2023: BGE Models are integrated into **Langchain**, you can use it like [this](#using-langchain); C-MTEB **leaderboard** is [available](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: Release base-scale and small-scale models, **best performance among the models of the same size 🤗**  
- 08/02/2023: Release `bge-large-*`(short for BAAI General Embedding) Models, **rank 1st on MTEB and C-MTEB benchmark!** :tada: :tada:   
- 08/01/2023: We release the [Chinese Massive Text Embedding Benchmark](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), consisting of 31 test dataset.  
  

</details>



## Projects

### BGE-M3([Paper](https://arxiv.org/pdf/2402.03216.pdf), [Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3))
In this project, we introduce BGE-M3, the first embedding model which supports multiple retrieval modes、multilingual and multi-granularity retrieval.
- Multi-Functionality: It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval. 
- Multi-Linguality: It can support more than 100 working languages. 
- Multi-Granularity: It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens. 

We propose a novel self-knowledge distillation approach to improve the performance of single retrieval mode. 
We optimize the batching strategy, enabling a large batch size, which can used simply when fine-tuning with long text or large language model. 
We also construct a dataset for document retrieval and propose a simple strategy to improve the ability to model long text.
**The training code and fine-tuning data will be open-sourced in the near future.**

### [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual)
In this project, we introduce Visualized-BGE, which integrating image token embedding into the BGE Text Embedding framework. Visualized-BGE can be used for various hybrid modal retrieval tasks, such as Multi-Modal Knowledge Retrieval, Composed Image Retrieval, and Knowledge Retrieval with Multi-Modal Queries.

Our model delivers outstanding zero-shot performance across multiple hybrid modal retrieval tasks. It can also serve as a base model for downstream fine-tuning for hybrid modal retrieval tasks.

### [LongLLM QLoRA](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)
We extend the context length of Llama-3-8B-Instruct from 8K to 80K via QLoRA fine-tuning. The entire training cycle is super efficient, which takes 8 hours on one 8xA800 (80G) GPU machine. The resulted model exhibits superior performances across a broad range of evaluation tasks, such as NIHS, topic retrieval, and long-context language understanding; meanwhile, it also well preserves the original capability over short contexts. The dramatic context extension is mainly attributed to merely 3.5K synthetic data generated by GPT-4, which indicates the LLMs' inherent (yet largely underestimated) potential to extend its original context length. In fact, the context length could be extended far beyond 80K with more computing resources.

### [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)

The utilization of long contexts poses a big challenge for large language models due to their limited context window length.
Activation Beacon condenses LLM's raw activations into more compact forms such that it can perceive a much longer context with a limited context window. 
It is an effective, efficient, compatible, and low-cost (training) method to extend the context length of LLM.
More details please refer to our [paper](https://arxiv.org/abs/2401.03462) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon).


### [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)

Model merging has been used to improve the performance of single model. 
We find this method is also useful for large language models and dense embedding model, 
and design the LM-Cocktail strategy which automatically merges fine-tuned models and base model using a simple function to compute merging weights.
LM-Cocktail can be used to improve the performance on target domain without decrease 
the general capabilities beyond target domain.
It also can be used to generate a model for new tasks without fine-tuning.
You can use it to merge the LLMs (e.g., Llama) or embedding models.
More details please refer to our report: [LM-Cocktail](https://arxiv.org/abs/2311.13534) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail).



### [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) 

LLM Embedder is fine-tuned based on the feedback from LLMs. 
It can support the retrieval augmentation needs of large language models, including knowledge retrieval, memory retrieval, example retrieval, and tool retrieval. 
It is fine-tuned over 6 tasks: Question Answering, Conversational Search, Long Conversation, 
Long-Range Language Modeling, In-Context Learning, and Tool Learning.
For more details please refer to [report](https://arxiv.org/abs/2310.07554) and [./FlagEmbedding/llm_embedder/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)


### [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)

Cross-encoder will perform full-attention over the input pair,
which is more accurate than embedding model (i.e., bi-encoder) but more time-consuming than embedding model.
Therefore, it can be used to re-rank the top-k documents returned by embedding model.
We train the cross-encoder on a multilingual pair data, 
The data format is the same as embedding model, so you can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker). 
For more details please refer to [./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)



We provide a new version of the cross-encoder that supports more languages and longer lengths. The data format is similar to our embedding models, but now includes prompt data for fine-tuning and inference. You can perform inference using specific layers or using the entire layers. You can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker#fine-tune).
For more details please refer to [./FlagEmbedding/llm_reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker).

### [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding) 

BGE embedding is a general Embedding Model. We pre-train the models using [retromae](https://github.com/staoxiao/RetroMAE) and train them on large-scale pair data using contrastive learning. 
**You can fine-tune the embedding model on your data following our [examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune).**
We also provide a [pre-train example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain).
Note that the goal of pre-training is to reconstruct the text, and the pre-trained model cannot be used for similarity calculation directly, it needs to be fine-tuned.
Refer to our [report: c-pack](https://arxiv.org/pdf/2309.07597.pdf) and [code](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) for more details.

**BGE uses the last hidden state of `[cls]` as the sentence embedding: `sentence_embeddings = model_output[0][:, 0]`. If you use mean pooling, there will be a significant decrease in performance.**


### [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
A benchmark for chinese text embedding. This benchmark has been merged into MTEB. 
Refer to our [report: c-pack](https://arxiv.org/pdf/2309.07597.pdf) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) for more details.


## Model List

`bge` is short for `BAAI general embedding`.

| Model                                                                     | Language |                                                                                                                                                                                             |                                                             Description                                                             |                                query instruction for retrieval                                 |
|:--------------------------------------------------------------------------|:--------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)                   |    Multilingual     |    [Inference](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3#usage) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3)    | Multi-Functionality(dense retrieval, sparse retrieval, multi-vector(colbert)), Multi-Linguality, and Multi-Granularity(8192 tokens) |  |
| [LM-Cocktail](https://huggingface.co/Shitao)                   |   English |                                                                                                                                                                                             |                     fine-tuned models (Llama and BGE) which can be used to reproduce the results of LM-Cocktail                     |  |
| [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)             |   English | [Inference](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |                         a unified embedding model to support diverse retrieval augmentation needs for LLMs                          | See [README](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |   Chinese and English |                                    [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker)                                    |                                   a cross-encoder model which is more accurate but less efficient                                   |                                                                                                |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)   |   Chinese and English |                                    [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker)                                    |                                   a cross-encoder model which is more accurate but less efficient                                   |                                                                                                |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)   |   English |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                      version 1.5 with more reasonable similarity distribution                                       |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)     |   English |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                      version 1.5 with more reasonable similarity distribution                                       |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)   |   English |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                      version 1.5 with more reasonable similarity distribution                                       |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)   |   Chinese |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                      version 1.5 with more reasonable similarity distribution                                       |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)     |   Chinese |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                      version 1.5 with more reasonable similarity distribution                                       |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)   |   Chinese |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                      version 1.5 with more reasonable similarity distribution                                       |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en)             |   English |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                             Embedding Model which map text into vector                                              |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en)               |   English |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                    a base-scale model but with similar ability to `bge-large-en`                                    |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en)             |   English |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                        a small-scale model but with competitive performance                                         |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)             |   Chinese |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                             Embedding Model which map text into vector                                              |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)               |   Chinese |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                    a base-scale model but with similar ability to `bge-large-zh`                                    |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh)             |   Chinese |                                [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune)                                 |                                        a small-scale model but with competitive performance                                         |                                     `为这个句子生成表示以用于检索相关文章：`                                      |




### Contributors:

<a href="https://github.com/FlagOpen/FlagEmbedding/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FlagOpen/FlagEmbedding" />
</a>




## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{bge_m3,
  title={BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
  author={Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
  year={2023},
  eprint={2309.07597},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{cocktail,
      title={LM-Cocktail: Resilient Tuning of Language Models via Model Merging}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Xingrun Xing},
      year={2023},
      eprint={2311.13534},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{llm_embedder,
      title={Retrieve Anything To Augment Large Language Models}, 
      author={Peitian Zhang and Shitao Xiao and Zheng Liu and Zhicheng Dou and Jian-Yun Nie},
      year={2023},
      eprint={2310.07554},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}

@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlagEmbedding is licensed under the [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). 

