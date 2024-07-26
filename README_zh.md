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


FlagEmbedding专注于检索增强llm领域，目前包括以下项目:

- **Long-Context LLM**: [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon), [LongLLM QLoRA](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)
- **Fine-tuning of LM** : [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail)
- **Embedding Model**: [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual), [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3), [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder), [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding)
- **Reranker Model**: [llm rerankers](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker), [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)
- **Benchmark**: [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB), [AIR-Bench](https://github.com/AIR-Bench/AIR-Bench), [MLVU](https://github.com/JUNJIE99/MLVU)

## 更新
- 6/7/2024: 发布首个专为长视频理解设计的全面评测基准[MLVU](https://github.com/JUNJIE99/MLVU)。MLVU拥有丰富的视频时长范围，多样化的视频来源，以及多个专为长视频理解设计的评估任务。🔥
- 5/21/2024：联合 Jina AI、Zilliz、HuggingFace 等机构发布评测基准 [AIR-Bench](https://github.com/AIR-Bench/AIR-Bench)，针对检索任务和 RAG 场景设计。AIR-Bench 首次提出在检索任务中使用 LLMs 自动化生产评估数据，避免模型过拟合测试数据。AIR-Bench 不需要人工参与标注数据，因而可以更灵活覆盖更多垂直领域和不同语种。同时 AIR-Bench 会定期进行更新从而满足社区不断变化的评测需求。[Leaderboard](https://huggingface.co/spaces/AIR-Bench/leaderboard) :fire:
- 4/30/2024: 发布[Llama-3-8B-Instruct-80K-QLoRA](https://huggingface.co/namespace-Pt/Llama-3-8B-Instruct-80K-QLoRA), 其通过在少量合成的长文本数据上的QLoRA训练，有效地将Llama-3-8B-Instruct的上下文长度从8K扩展到80K。详见[代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora) :fire:
- 3/18/2024: 发布新的[rerankers](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker), 拥有更好的性能同时支持多语言和长文本。 :fire:
- 3/18/2024: 发布[Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual)，该项目通过引入image token embedding赋予BGE视觉编码能力。Visualized-BGE可以对混合图文数据进行编码，用于广泛的混合模态检索任务。 :fire:
- 1/30/2024: 发布**BGE-M3**, 第一个具有多功能、多语言和多粒度特性的文本检索模型，高效支持多语言（100+语言）、长文本（至多8192长度的输入文本）、和混合检索（稠密、稀疏、多向量）。 详见[report](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/BGE_M3.pdf)和[代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3)  :fire:
- 1/9/2024: 发布[Activation-Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon), 一个有效、高效、兼容、低成本（训练）的扩展大预言模型上下文长度的方法。[技术报告](https://arxiv.org/abs/2401.03462) 
- 12/24/2023: 发布**LLaRA**, 一个基于LLaMA-7B的稠密检索模型, MS MARCO与BEIR上取得了迄今最好的实验结果. 模型与代码将会陆续开源. 敬请关注. [技术报告](https://arxiv.org/abs/2312.15503) 
- 11/23/2023: 发布[LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/LM_Cocktail), 一种通过模型融合在微调时保持原有模型通用能力的方法. [技术报告](https://arxiv.org/abs/2311.13534) 
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

### BGE-M3([Paper](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/BGE_M3.pdf), [Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3))
在这个项目中，我们发布了BGE-M3，它是第一个具有多功能、多语言和多粒度特性的文本检索模型。
- 多功能:可以同时执行三种检索功能：单向量检索、多向量检索和稀疏检索。
- 多语言:支持100多种工作语言。
- 多粒度:它能够处理不同粒度的输入，从短句子到长达8192个词汇的长文档。  

在本项目中，为了提高单一检索模式的性能，提出了一种新的自知识蒸馏方法。 
我们优化了批处理策略，支持大批处理大小，这可以在对长文本或大型语言模型进行向量微调时简单使用。
我们还构建了一个用于文档检索的数据集，并提出了一个简单的策略来提高长文本的建模能力。
**训练代码和微调数据将在不久的将来开源。**

### [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual)
在这个项目中，我们发布了Visualized-BGE。
通过引入image token embedding，Visualized-BGE可以被用来编码混合图文数据。它可以被应用在广泛的多模态检索任务中，包括但不限于：多模态知识检索，多模态查询的图像检索等。

### [LongLLM QLoRA](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)
我们通过 QLoRA 微调将 Llama-3-8B-Instruct 的上下文长度从 8K 扩展到 80K。 整个训练过程非常高效，在一台8xA800 (80G) GPU 机器上仅需要8个小时。 该模型在NIHS、主题检索和长上下文语言理解等广泛的评估任务中表现出卓越的性能； 同时，它在短上下文中也很好地保留了其原有的能力。 如此强大的长文本能力主要归因于GPT-4生成的仅3.5K合成数据，这表明LLM具有扩展其原始上下文的固有（但在很大程度上被低估）潜力。 事实上，一旦有更多的计算资源，该方法可以将上下文长度扩展更长。

### [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)

由于有限的上下文窗口长度，有效利用长上下文信息是对大型语言模型的一个巨大挑战。
Activation Beacon 将 LLM 的原始激活压缩为更紧凑的形式，以便它可以在有限的上下文窗口中感知更长的上下文。
它是一种有效、高效、兼容、低成本（训练）的延长LLM上下文长度的方法。
更多细节请参考[技术报告](https://arxiv.org/abs/2401.03462)和[代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon)。


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



我们提供了新版的交叉编码器，支持更多的语言以及更长的长度。使用的数据格式与向量模型类似，但是新增了prompt用于微调以及推理。您可以使用特定的层进行推理或使用完整的层进行推理，您可以根根据我们的[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker#fine-tune) 轻松地对其进行微调。
更多细节请参考[./FlagEmbedding/llm_reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/llm_reranker/README.md)

### [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/baai_general_embedding) 

BGE Embedding是一个通用向量模型。 我们使用[retromae](https://github.com/staoxiao/RetroMAE) 对模型进行预训练，再用对比学习在大规模成对数据上训练模型。
**你可以按照我们的[示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) 在本地数据上微调嵌入模型。**
我们还提供了一个[预训练示例](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain) 。
请注意，预训练的目标是重构文本，预训练后的模型无法直接用于相似度计算，需要进行微调之后才可以用于相似度计算。
更多关于bge的训练情况请参阅[论文](https://arxiv.org/pdf/2309.07597.pdf)和[代码](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md).

**注意BGE使用CLS的表征作为整个句子的表示，如果使用了错误的方式（如mean pooling)会导致效果很差。**


### [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)
中文向量榜单，已整合入MTEB中。更多细节参考 [论文](https://arxiv.org/pdf/2309.07597.pdf) 和[代码](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB).





## 模型列表
| Model                                                                     |      Language       | |              Description               | query instruction for retrieval [1] |
|:--------------------------------------------------------------------------|:-------------------:| :--------:|:--------------------------------------:|:--------:|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)                   |    Multilingual     | [推理](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3#usage) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3) | 多功能（向量检索，稀疏检索，多表征检索）、多语言、多粒度（最大长度8192） |  |
| [LM-Cocktail](https://huggingface.co/Shitao)                              |       English       |  | 微调的Llama和BGE模型，可以用来复现LM-Cocktail论文的结果  |  |
| [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)             |       English       | [推理](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |         专为大语言模型各种检索增强任务设计的向量模型         | 详见 [README](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | Chinese and English | [推理](#usage-for-reranker) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) |      交叉编码器模型，精度比向量模型更高但推理效率较低 [2]      |   |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)   | Chinese and English | [推理](#usage-for-reranker) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) |      交叉编码器模型，精度比向量模型更高但推理效率较低 [2]      |   |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)   |       English       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理             | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)     |       English       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理             | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)   |       English       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理             | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)   |       Chinese       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理             | `为这个句子生成表示以用于检索相关文章：`  |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)     |       Chinese       |  [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理             | `为这个句子生成表示以用于检索相关文章：`  |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)   |       Chinese       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            1.5版本，相似度分布更加合理             | `为这个句子生成表示以用于检索相关文章：`  |
| [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en)             |       English       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |             向量模型，将文本转换为向量              | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en)               |       English       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            base-scale 向量模型             | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en)             |       English       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            small-scale 向量模型            | `Represent this sentence for searching relevant passages: `  |
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)             |       Chinese       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |             向量模型，将文本转换为向量              | `为这个句子生成表示以用于检索相关文章：`  |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)               |       Chinese       |  [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            base-scale 向量模型             | `为这个句子生成表示以用于检索相关文章：`  |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh)             |       Chinese       | [推理](#usage-for-embedding-model) [微调](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |            small-scale 向量模型            | `为这个句子生成表示以用于检索相关文章：`  |




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



