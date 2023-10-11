# Fine-tuning

## Data
The following data format is universally used for training & evaluating retrievers and rerankers.

**We will release our training data soon.**

```python
# training
{
  "query": str,
  "pos": List[str],
  "neg": List[str],
  "pos_index": Optional[List[int]],         # Indices of the positives w.r.t. the corpus. When a global corpus is not available (e.g. long conversation), just ignore this field.
  "neg_index": Optional[List[int]],         # Indices of the negatives w.r.t. the corpus. When a global corpus is not available (e.g. long conversation), just ignore this field.
  "teacher_scores": Optional[List[float]],  # Scores from an LM or a reranker, used for distillation.
  "answers": Optional[List[str]],           # List of answers for the query, used for LM scoring.
}

# evaluation
{
  "query": str,
  "pos_index": Optional[List[int]],         # Indices of the positives w.r.t. corpus (retrieval) / w.r.t. keys (rerank). When there is no positives pre-defined (e.g. NQ), just ignore this field.
  "answers": Optional[List[str]],           # List of answers for computing NQ metrics.
  "key": Optional[List[str]],               # Collated retrieval results for the query / candidates to rank when there are no positives and negatives.
  "key_index": Optional[List[int]],         # Key indices w.r.t. the corpus when reranking and no positives & negatives.
}
```

## Retriever
There are several important arguments for training:
- `train_data`: required, one or a list of json files with the aforementioned formatting.
- `eval_data`: optional, one json file with the aforementioned formatting. If an `eval_data` is speficied, the trainer will automatically do evaluation on the `eval_data` every `save_steps`.
- `corpus`: optional, the global corpus where `positives` and `negatives` come from.

The meaning and usage of other arguments can be inspected from [code](../src/retrieval/args.py) or running `python run_dense.py --help` from command line.

### LLM-Embedder (Multi-Task Fine-Tune)
```bash
bash scripts/llm-embedder.sh
```

### Single Task Fine-Tune
Below we provide commands to fine-tune a retriever on a single task.

#### QA
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/nq/contrastive \
--train_data llm-embedder:qa/nq/train.json \
--eval_data llm-embedder:qa/nq/test.json \
--corpus llm-embedder:qa/nq/corpus.json \
--metrics nq \
--key_max_length 128 \
--query_max_length 32 \
--max_steps 2000
```

#### In-Context Learning
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/icl/distill \
--train_data llm-embedder:icl/icl/train.scored.llama2-chat.top20.json \
--max_steps 6000 \
--save_steps 6000 \
--add_instruction false \
--select_positive random \
--contrastive_weight 0 \
--stable_distill
```

#### Long-Range Language Modeling
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/lrlm/distill \
--train_data llm-embedder:lrlm/books3/train.128tok.scored.llama2-7b-chat.json llm-embedder:lrlm/arxiv/train.128tok.scored.llama2-7b-chat.json llm-embedder:lrlm/codeparrot/train.128tok.scored.llama2-7b-chat.json \
--select_positive teacher \
--select_negative random \
--teacher_scores_margin 0.1 \
--contrastive_weight 0 \
--teacher_temperature 0.1 \
--save_steps 4000 \
--max_steps 4000 \
--learning_rate 5e-6
```

#### Long Chat
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/msc/distill \
--train_data llm-embedder:chat/msc/train.scored.llama2-7b-chat.json \
--select_positive teacher \
--select_negative random \
--contrastive_weight 0 \
--teacher_temperature 0.1 \
--max_steps 4000 \
--save_steps 4000 \
--learning_rate 5e-6
```

#### Tool
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/tool/contrastive \
--train_data llm-embedder:tool/toolbench/train.hardneg.json \
--eval_data llm-embedder:tool/toolbench/test.json \
--corpus llm-embedder:tool/toolbench/corpus.json \
--save_steps 2000 \
--max_steps 2000 \
--key_template {text} \
--metrics ndcg
```

#### Conversation Search
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/qrecc/contrastive-concat-from-bm25-human \
--train_data llm-embedder:conversation/qrecc/train.concat.neg.bm25-from-human.json \
--metrics mrr ndcg \
--cutoffs 3 10 100 \
--max_steps 2000 \
--key_template '{text}'
```

### Mine Negatives
```bash
# BGE
torchrun --nproc_per_node=8 -m evaluation.eval_retrieval \
--eval_data llm-embedder:qa/nq/train.json \
--corpus llm-embedder:qa/nq/corpus.json \
--metrics mrr recall collate_neg \
--save_name bge

# BM25
torchrun --nproc_per_node 8 -m evaluation.eval_retrieval \
--retrieval_method bm25 \
--eval_data llm-embedder:qa/nq/train.json \
--corpus llm-embedder:qa/nq/corpus.json \
--metrics mrr recall collate_neg \
--save_name bm25
```

## LM Scoring
Score positives and negatives in `eval_data` with $p(o|q,k)$ where $o$ is the desired output, $q$ is the query, and $k$ is a key (could be positive or negative). This requires `answers` field in `train_data`.

```bash
torchrun --nproc_per_node=8 run_lm_score.py --eval_data llm-embedder:qa/msmarco/train.json
```
Results will be saved at `llm-embedder:qa/msmarco/train.scored.llama2-7b.json`

### 3-Iter Pipeline
We replicate the 3-iter pipeline for enhancing retriever's performance from [SimLM paper](https://arxiv.org/abs/2207.02578).

```bash
bash scripts/3iter-msmarco.sh
bash scripts/3iter-nq.sh
```

## Note
- `transformers==4.30.0` raises error when using deepspeed schedulerconfig
  - modify line `1750` in `trainer.py`
  ```python
    if use_accelerator_prepare:
        # NOTE: fix bug in transformers 4.30.0
        # model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.model.train()
        if hasattr(self.lr_scheduler, "step"):
            if self.use_apex:
                model = self.accelerator.prepare(self.model)
            else:
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
  ```