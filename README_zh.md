<h1 align="center">FlagEmbedding</h1>
<p align="center">
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made with-Python-purple">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <a href="https://huggingface.co/C-MTEB">
        <img alt="License" src="https://img.shields.io/badge/C_MTEB-🤗-yellow">
    </a>
    <a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding">
        <img alt="License" src="https://img.shields.io/badge/universal embedding-1.1-red">
    </a>
</p>

<h4 align="center">
    <p>
        <a href=#更新>更新</a> |
        <a href="#项目">项目</a> |
        <a href="#模型列表">模型列表</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

[English](README.md) | [中文](README_zh.md)


<span style="#FF69B4;"> **Hiring:** 我们正在招聘NLP研究人员和实习生，专注于检索增强大模型领域。如果您感兴趣，请随时通过电子邮件与我们联系：zhengliu1026@gmail.com.</span>

FlagEmbedding专注于检索增强llm领域，目前包括以下项目:

- **Fine-tuning of LM** : [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)
- **Dense Retrieval**: [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding), [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
- **Reranker Model**: [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)


## 更新

- 11/23/2023: Release [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail), 一种通过模型融合在微调时保持原有模型通用能力的方法. [技术报告](https://arxiv.org/abs/2311.13534) :fire:
- 10/12/2023: 发布 [LLM-Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), 专为大语言模型**各种检索增强任务设计**的英文向量模型。[技术报告](https://arxiv.org/pdf/2310.07554.pdf) 
- 09/15/2023: 发布 [技术报告](https://arxiv.org/pdf/2309.07597.pdf) 和 [数据集](https://data.baai.ac.cn/details/BAAI-MTP).
- 09/12/2023: 更新：
    - **新增重排模型**：开源交叉编码器模型bge-reranker，具有比向量模型更强大的排序能力。非常建议使用或者微调它来重新排序向量模型返回的top-k文档，提高最终结果的相关性。
    - **更新向量模型**：发布bge-*-v1.5向量模型，缓解相似度分布问题，提升无指令情况下的检索能力（但检索任务仍建议使用指令）
- 09/07/2023: 更新[微调代码](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md): 增加难负样本挖掘脚本，增加指令参数方便在微调中添加指令.
- 08/09/2023: BGE模型整合入Langchain, 可以在langchain中非常简单的[使用它](#using-langchain); C-MTEB中文榜单已[在线更新](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: 发布更小的模型(base, small), **在同尺寸模型中取得最好的性能！ 🤗**
- 08/02/2023: :tada: :tada: 发布中英文向量模型BGE(BAAI General Embedding的缩写), **在MTEB和C-MTEB榜单上取得最好的性能** 
- 08/01/2023: 发布大规模中文文本向量[评测榜单](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), 其包括31个测试任务.   





## 项目

### [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)

模型合并被用于提高单模型的性能。
我们发现这种方法对大型语言模型和文本向量模型也很有用， 并设计了”语言模型鸡尾酒“方案，其自动计算融合比例去融合基础模型和微调模型。
利用LM-Cocktail可以缓解灾难性遗忘问题，即在不降低通用性能的情况下提高目标任务性能。
通过构造少量数据样例，它还可以用于为新任务生成模型，而无需进行微调。
它可以被使用来合并生成模型或向量模型。
更多细节请参考[技术报告](https://arxiv.org/abs/2311.13534)和[代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)。


### [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) 

LLM-Embedder向量模型是根据LLM的反馈进行微调的。
它可以支持大型语言模型的检索增强需求，包括知识检索、记忆检索、示例检索和工具检索。
它在6个任务上进行了微调:问题回答，对话搜索，长对话，
长文本建模、上下文学习和工具学习。
更多细节请参考[./FlagEmbedding/llm_embedder/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder)


### [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)

交叉编码器将对查询和答案实时计算相关性分数，这比向量模型(即双编码器)更准确，但比向量模型更耗时。
因此，它可以用来对嵌入模型返回的前k个文档重新排序。
我们在多语言数据上训练了交叉编码器，数据格式与向量模型相同，因此您可以根据我们的[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) 轻松地对其进行微调。
更多细节请参考[./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/reranker/README.md)



### [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding) 

BGE Embedding是一个通用向量模型。 我们使用[retromae](https://github.com/staoxiao/RetroMAE) 对模型进行预训练，再用对比学习在大规模成对数据上训练模型。
**你可以按照我们的[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) 在本地数据上微调嵌入模型。**
我们还提供了一个[预训练示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain) 。
请注意，预训练的目标是重构文本，预训练后的模型无法直接用于相似度计算，需要进行微调之后才可以用于相似度计算。
更多关于bge的训练情况请参阅[baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) ，





## 模型列表
|              Model              | Language | |              Description              | query instruction for retrieval [1] |
|:-------------------------------|:--------:| :--------:|:-------------------------------------:|:--------:|
| [LM-Cocktail](https://huggingface.co/Shitao)                   |   English |  | 微调的Llama和BGE模型，可以用来复现LM-Cocktail论文的结果 |  |
|  [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)  |   English | [推理](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |        专为大语言模型各种检索增强任务设计的向量模型         | 详见 [README](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |
|  [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |   Chinese and English | [推理](#usage-for-reranker) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) |     交叉编码器模型，精度比向量模型更高但推理效率较低 [2]      |   |
|  [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |   Chinese and English | [推理](#usage-for-reranker) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) |     交叉编码器模型，精度比向量模型更高但推理效率较低 [2]      |   |
|  [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理            | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理            | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理            | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理            | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |   Chinese |  [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理            | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理            | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |             向量模型，将文本转换为向量             | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            base-scale 向量模型            | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |           small-scale 向量模型            | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |             向量模型，将文本转换为向量             | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            base-scale 向量模型            | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |           small-scale 向量模型            | `为这个句子生成表示以用于检索相关文章：`  |




## Contributors:

<a href="https://github.com/FlagOpen/FlagEmbedding/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=FlagOpen/FlagEmbedding" />
</a>
 


## Citation

如果您觉得我们的工作有所帮助，请考虑点个星 :star: 和引用以下论文:
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
FlagEmbedding基于[MIT License](LICENSE)开源协议。



