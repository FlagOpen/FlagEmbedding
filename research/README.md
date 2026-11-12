# Research

### BGE-M3 ([Paper](https://arxiv.org/pdf/2402.03216.pdf), [Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_M3))


In this project, we introduce BGE-M3, the first embedding model which supports:

- **Multi-Functionality**: It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval.
- **Multi-Linguality**: It can support more than 100 working languages.
- **Multi-Granularity**: It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens.

**The training code and fine-tuning data will be open-sourced in the near future.**

### [Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge)

In this project, we introduce Visualized-BGE, which integrating image token embedding into the BGE Text Embedding framework. Visualized-BGE can be used for various hybrid modal retrieval tasks, such as Multi-Modal Knowledge Retrieval, Composed Image Retrieval, and Knowledge Retrieval with Multi-Modal Queries.

Our model delivers outstanding zero-shot performance across multiple hybrid modal retrieval tasks. It can also serve as a base model for downstream fine-tuning for hybrid modal retrieval tasks.



### [LongLLM QLoRA](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/longllm_qlora)

We extend the context length of Llama-3-8B-Instruct from 8K to 80K via QLoRA fine-tuning. The entire training cycle is super efficient, which takes 8 hours on one 8xA800 (80G) GPU machine (the context length can go far beyond 80k with more computing resources). The resulted model exhibits superior performances across a broad range of evaluation tasks, such as NIHS, topic retrieval, and long-context language understanding; meanwhile, it also well preserves the original capability over short contexts.


### [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon)

The utilization of long contexts poses a big challenge for large language models due to their limited context window length.
Activation Beacon condenses LLM's raw activations into more compact forms such that it can perceive a much longer context with a limited context window. 
It is an effective, efficient, compatible, and low-cost (training) method to extend the context length of LLM.
More details please refer to our [paper](https://arxiv.org/abs/2401.03462) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon).


### [LM-Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/LM_Cocktail)

LM-Cocktail automatically merges fine-tuned models and base model using a simple function to compute merging weights.
LM-Cocktail can be used to improve the performance on target domain without decrease the general capabilities beyond target domain, 
as well as generate a model for new tasks without fine-tuning.
You can use it to merge the LLMs (e.g., Llama) or embedding models.
More details please refer to our report: [LM-Cocktail](https://arxiv.org/abs/2311.13534) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/LM_Cocktail).



### [LLM Embedder](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder) 

LLM Embedder is fine-tuned based on the feedback from LLMs. 
It supports the retrieval augmentation needs of large language models, including knowledge retrieval, memory retrieval, example retrieval, and tool retrieval. 
It is fine-tuned over 6 tasks: Question Answering, Conversational Search, Long Conversation, 
Long-Range Language Modeling, In-Context Learning, and Tool Learning.
For more details please refer to [report](https://arxiv.org/abs/2310.07554) and [./llm_embedder/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_embedder)


### [BGE Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/reranker)

Cross-encoder will perform full-attention over the input pair,
which is more accurate than embedding model (i.e., bi-encoder) but more time-consuming than embedding model.
Therefore, it can be used to re-rank the top-k documents returned by embedding model.
We train the cross-encoder on a multilingual pair data, 
The data format is the same as embedding model, so you can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker). 
For more details please refer to [./reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/reranker)


### [LLM Reranker](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_reranker) 

We provide a new version of the cross-encoder that supports more languages and longer lengths. The data format is similar to our embedding models, but now includes prompt data for fine-tuning and inference. You can perform inference using specific layers or using the entire layers. You can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_reranker#fine-tune).
For more details please refer to [./llm_reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/llm_reranker).

### [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/baai_general_embedding) 

BGE embedding is a general Embedding Model. We pre-train the models using [retromae](https://github.com/staoxiao/RetroMAE) and train them on large-scale pair data using contrastive learning. 
**You can fine-tune the embedding model on your data following our [examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune/embedder).**
We also provide a [pre-train example](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/old-examples/pretrain).
Note that the goal of pre-training is to reconstruct the text, and the pre-trained model cannot be used for similarity calculation directly, it needs to be fine-tuned.
Refer to our [report: c-pack](https://arxiv.org/pdf/2309.07597.pdf) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/baai_general_embedding) for more details.

**BGE uses the last hidden state of `[cls]` as the sentence embedding: `sentence_embeddings = model_output[0][:, 0]`. If you use mean pooling, there will be a significant decrease in performance.**


### [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB)

A benchmark for chinese text embedding. This benchmark has been merged into MTEB. 
Refer to our [report: c-pack](https://arxiv.org/pdf/2309.07597.pdf) and [code](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB) for more details.
