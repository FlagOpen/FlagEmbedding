<div align="center">
<h1> BGE-Reasoner: Towards End-to-End Reasoning-Intensive Information Retrieval </h1>
</div>

## Introduction

We introduce **BGE-Reasoner**, an end-to-end reasoning-intensive information retrieval framework. BGE-Reasoner is characterized by three key features:

1. **End-to-end**: It comprises three core components in IR—**BGE-Reasoner-Rewriter**, **BGE-Reasoner-Embed**, and **BGE-Reasoner-Reranker**—covering the entire retrieval pipeline, from query rewriting and retrieval to reranking for reasoning-intensive tasks.
2. **Excellent performance**: **BGE-Reasoner** achieves **state-of-the-art (SOTA)** performance on [BRIGHT](https://brightbenchmark.github.io/), a reasoning-intensive information retrieval benchmark, with an **nDCG@10 of 43.8** across 12 datasets, outperforming the previous SOTA by +2.2 points (41.6 from [DIVER](https://arxiv.org/pdf/2508.07995), Aug 12, 2025).
3. **Open-source resources**: We will release the code, model checkpoints, training data, and evaluation scripts to facilitate future research on reasoning-intensive information retrieval. Please stay tuned!


## Open-source resources

| Resource Type      | Name                  | Hugging Face Link |
| ------------------ | --------------------- | ----------- |
| Model              | BGE-Reasoner-Rewriter | [🤗]() (TBA)     |
| Model              | BGE-Reasoner-Embed    | [🤗]() (TBA)     |
| Model              | BGE-Reasoner-Reranker | [🤗]() (TBA)     |
| Training Data      | BGE-Reasoner-Data     | [🤗]() (TBA)     |
| Evaluation Scripts | -                     | (TBA)             |


## Performance

**BGE-Reasoner** achieves SOTA performance on the **BRIGHT** benchmark with the following pipeline:

1. **Query Rewrite**: **BGE-Reasoner-Rewriter** rewrites the original query to a more reasoning-friendly form.
2. **Retrieval**: **BGE-Reasoner-Embed** and **BM25** each retrieve the top-2000 documents from the corpus using the rewritten query. Hybrid retrieval combines the results from both methods, with a weight of 0.75 for BGE-Reasoner-Embed and 0.25 for BM25 (after min-max normalization).
3. **Reranking**: **BGE-Reasoner-Reranker** reranks the top-100 documents from the previous step. Hybrid retrieval combines the reranked results with those from the previous step, assigning a weight of 0.5 to each method.

Our work is still in progress. We temporarily report the preview-version results of **BGE-Reasoner** on the **BRIGHT** benchmark as shown below. Detailed results for each component, along with additional technical details, will be released soon.

|                                                              |          | StackExchange |            |           |          |          |            |          | Coding    |          | Theorem-based |            |            |
| ------------------------------------------------------------ | -------- | ------------- | ---------- | --------- | -------- | -------- | ---------- | -------- | --------- | -------- | ------------- | ---------- | ---------- |
|                                                              | **Avg.** | **Bio.**      | **Earth.** | **Econ.** | **Psy.** | **Rob.** | **Stack.** | **Sus.** | **Leet.** | **Pony** | **AoPS**      | **TheoQ.** | **TheoT.** |
| [ReasonIR with QwenRerank](https://arxiv.org/pdf/2504.20595) | 36.9     | 58.2          | 53.2       | 32.0      | 43.6     | 28.8     | 37.6       | 36.0     | 33.2      | 34.8     | 7.9           | 32.6       | 45.0       |
| [ReasonIR with Rank-R1](https://huggingface.co/ielabgroup/Rank-R1-32B-v0.2) | 40.0     | 64.4          | 60.1       | 38.3      | 52.2     | 30.7     | 40.6       | **46.7** | 33.3      | 17.4     | 10.1          | 38.6       | 47.7       |
| [XRR2](https://github.com/jataware/XRR2)                     | 40.3     | 63.1          | 55.4       | **38.5**  | 52.9     | **37.1** | 38.2       | 44.6     | 21.9      | 35.0     | 15.7          | 34.4       | 46.2       |
| [RaDeR with ReasonRank](http://arxiv.org/pdf/2508.07050)     | 40.8     | 62.7          | 55.5       | 36.7      | **54.6** | 35.7     | 38.0       | 44.8     | 29.5      | 25.6     | 14.4          | 42.0       | 50.1       |
| [DIVER](https://arxiv.org/pdf/2508.07995)                    | 41.6     | 62.2          | 58.7       | 34.4      | 52.9     | 35.6     | 36.5       | 42.9     | **38.9**  | 25.4     | **18.3**      | 40.0       | **53.1**   |
| **BGE-Reasoner (Ours)**                                      | **43.8** | **67.3**      | **64.4**   | 38.2      | 50.8     | 36.9     | **42.4**   | 43.0     | 29.4      | **43.8** | 14.5          | **42.5**   | 52.2       |

## Citation

TBA
