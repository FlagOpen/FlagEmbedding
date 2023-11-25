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

<span style="#FF69B4;"> **Hiring:** We're seeking experienced NLP researchers and intern students focusing on dense retrieval and retrieval-augmented LLMs. If you're interested, please feel free to reach out to us via email at zhengliu1026@gmail.com.</span>

FlagEmbedding focus on retrieval-augmented LLMs, consisting of following projects currently:

- **Fine-tuning of LM** : [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)
- **Dense Retrieval**: [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding), [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- **Reranker Model**: [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)


## News 

- 11/23/2023: Release [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/LM_Cocktail), a method to maintain general ability during fine-tuning by merging multiple models. [Paper](https://arxiv.org/abs/2311.13534) :fire:  
- 10/12/2023: Release [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), a unified embedding model to support diverse retrieval augmentation needs for LLMs. [Paper](https://arxiv.org/pdf/2310.07554.pdf)
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

### [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)

The pre-trained language models are continually fine-tuned to better support downstream
applications. However, this operation may result in significant performance degeneration on
general tasks beyond the targeted domain. To overcome this problem, we propose a novel
method which enables the fine-tuned model
to stay resilient in general perspectives: LM-Cocktail. 
LM-Cocktail can achieve
a strong empirical performance in the whole
scope of general tasks while preserving a superior capacity in its targeted domain. It also can be used to generate a model for new tasks without fine-tuning.
You can use it to merge the LLMs (e.g., Llama) or embedding models.
More details please refer to our paper: [LM-Cocktail](https://arxiv.org/abs/2311.13534) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail).



### [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) 

LLM Embedder is fine-tuned based on the feedback from LLMs. 
It can support the retrieval augmentation needs of large language models, including knowledge retrieval, memory retrieval, examplar retrieval, and tool retrieval. 
It is fine-tuned over 6 tasks: Question Answering, Conversational Search, Long Conversation, 
Long-Range Language Modeling, In-Context Learning, and Tool Learning.
For more details please refer to [./FlagEmbedding/llm_embedder/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)


### [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)

Cross-encoder will perform full-attention over the input pair, 
which is more accurate than embedding model (i.e., bi-encoder) but more time-consuming than embedding model.
Therefore, it can be used to re-rank the top-k documents returned by embedding model.
We train the cross-encoder on a multilingual pair data, 
The data format is the same as embedding model, so you can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker). 
For more details please refer to [./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)


### [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding) 

BGE embedding is a general Embedding Model. We pre-train the models using [retromae](https://github.com/staoxiao/RetroMAE) and train them on large-scale pair data using contrastive learning. 
**You can fine-tune the embedding model on your data following our [examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune).**
We also provide a [pre-train example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain).
Note that the goal of pre-training is to reconstruct the text, and the pre-trained model cannot be used for similarity calculation directly, it needs to be fine-tuned.
For more training details for bge see [baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md).



## Model List

`bge` is short for `BAAI general embedding`.

| Model                                                                     | Language | |                                                Description                                                |                                query instruction for retrieval                                 |
|:--------------------------------------------------------------------------|:--------:| :--------:|:---------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| [LM-Cocktail](https://huggingface.co/Shitao)                   |   English |  |        fine-tuned models (Llama and BGE) which can be used to reproduce the results of LM-Cocktail        |  |
| [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)             |   English | [Inference](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |            a unified embedding model to support diverse retrieval augmentation needs for LLMs             | See [README](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) |                      a cross-encoder model which is more accurate but less efficient                      |                                                                                                |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)   |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) |                      a cross-encoder model which is more accurate but less efficient                      |                                                                                                |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)   |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                         version 1.5 with more reasonable similarity distribution                          |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)     |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                         version 1.5 with more reasonable similarity distribution                          |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)   |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                         version 1.5 with more reasonable similarity distribution                          |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)   |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                         version 1.5 with more reasonable similarity distribution                          |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)     |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                         version 1.5 with more reasonable similarity distribution                          |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)   |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                         version 1.5 with more reasonable similarity distribution                          |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en)             |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |        :trophy: rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard        |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en)               |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                       a base-scale model but with similar ability to `bge-large-en`                       |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en)             |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                           a small-scale model but with competitive performance                            |                  `Represent this sentence for searching relevant passages: `                   |
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)             |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)               |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                       a base-scale model but with similar ability to `bge-large-zh`                       |                                     `为这个句子生成表示以用于检索相关文章：`                                      |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh)             |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |                           a small-scale model but with competitive performance                            |                                     `为这个句子生成表示以用于检索相关文章：`                                      |




### Contributors:

<a href="https://github.com/FlagOpen/FlagEmbedding/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FlagOpen/FlagEmbedding" />
</a>




## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
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


