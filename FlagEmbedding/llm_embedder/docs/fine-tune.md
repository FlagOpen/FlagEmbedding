# Fine-tuning

## Environment
It is recommended that you create a new environment:
```
cd FlagEmbedding/llm_embedder

conda env create -f environment.yaml --name llm-embedder
conda activate llm-embedder
```

## Data
The data for fine-tuning & evaluation can be downloaded [here](https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/llm-embedder.tar.gz). You should untar the file at anywhere you prefer, e.g. `/data`, which results in a folder `/data/llm-embedder`:
```bash
tar -xzvf llm-embedder-eval.tar.gz -C /data
```

The corpus of QReCC for conversational search is too large (54M passages), we separately upload it to huggingface datasets [namespace-Pt/qrecc-corpus](https://huggingface.co/datasets/namespace-Pt/qrecc-corpus). To evaluate the performance on conversational search, you should load it and save it as json file in the `qrecc` folder:
```python
import datasets
# load dataset
qrecc_corpus = datasets.load_dataset("namespace-Pt/qrecc-corpus", split="train")
# save to jsonline format in YOUR data folder
qrecc_corpus.to_json("/data/llm-embedder/convsearch/qrecc/corpus.json", force_ascii=False, lines=True, orient="records")
```

The data formats for training and evaluation are as follows:

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
Below are several important arguments for training. The meaning and usage of other arguments can be inspected from [code](../src/retrieval/args.py) or running `python run_dense.py --help` from command line.
- `train_data`: required, one or a list of json files with the aforementioned formatting.
- `eval_data`: optional, one json file with the aforementioned formatting. If an `eval_data` is speficied, the trainer will automatically do evaluation on the `eval_data`.
- `corpus`: optional, the global corpus where `positives`.

**IMPORTANT NOTE**
- For any path specified for `train_data`, `eval_data`, and `corpus`: if it is prefixed with `llm-embedder`, it will be solved to the relative path against [`data_root`](../src/retrieval/args.py). *Note that you can modify the default value of `data_root`, so that you don't need to type it for each command.*
- During fine-tuning, we save the output model in the `huggingface transformers`🤗 format. To use it from `sentence_transformers`, you should convert it to `sentence_transformers` checkpoint in advance:
  ```bash
  python scripts/ours2st.py --encoder data/outputs/your-output-dir/encoder
  ```
  Then everything is the same as described in [README](../README.md).

### LLM-Embedder (Multi-Task Fine-Tune)
```bash
# Remember to modify the data_root to your data root in the script :)
bash scripts/llm-embedder.sh
```

### Single Task Fine-Tune
Below we provide commands to fine-tune a retriever on a single task.

#### QA
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/nq \
--train_data llm-embedder:qa/nq/train.json \
--eval_data llm-embedder:qa/nq/test.json \
--corpus llm-embedder:qa/nq/corpus.json \
--metrics nq \
--key_max_length 128 \
--query_max_length 32 \
--max_steps 2000 \
--contrastive_weight 0 \
--stable_distill \
--data_root /data/llm-embedder
```

#### In-Context Learning
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/icl \
--train_data llm-embedder:icl/icl/train.json \
--max_steps 6000 \
--save_steps 6000 \
--select_positive random \
--contrastive_weight 0 \
--stable_distill \
--data_root /data/llm-embedder
```

#### Long-Range Language Modeling
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/lrlm \
--train_data llm-embedder:lrlm/books3/train.json llm-embedder:lrlm/arxiv/train.json llm-embedder:lrlm/codeparrot/train.json \
--max_steps 4000 \
--save_steps 4000 \
--select_positive teacher \
--teacher_scores_margin 0.1 \
--contrastive_weight 0 \
--teacher_temperature 0.1 \
--data_root /data/llm-embedder
```

#### Long Chat
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/msc \
--train_data llm-embedder:chat/msc/train.json \
--max_steps 4000 \
--save_steps 4000 \
--select_positive teacher \
--select_negative random \
--contrastive_weight 0 \
--teacher_temperature 0.1 \
--data_root /data/llm-embedder
```

#### Tool
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/tool \
--train_data llm-embedder:tool/toolbench/train.json \
--eval_data llm-embedder:tool/toolbench/test.json \
--corpus llm-embedder:tool/toolbench/corpus.json \
--key_template {text} \
--metrics ndcg \
--max_steps 2000 \
--data_root /data/llm-embedder
```

#### Conversation Search
```bash
torchrun --nproc_per_node=8 run_dense.py \
--output_dir data/outputs/qrecc \
--train_data llm-embedder:conversation/qrecc/train.concat.json \
--eval_data llm-embedder:conversation/qrecc/test.concat.json \
--corpus llm-embedder:conversation/qrecc/corpus.json \
--key_template '{text}' \
--metrics mrr ndcg \
--cutoffs 3 10 100 \
--max_steps 2000 \
--data_root /data/llm-embedder
```

### Mine Negatives
```bash
# BGE (the result will be saved at llm-embedder:qa/nq/train.neg.bge.json)
torchrun --nproc_per_node=8 -m evaluation.eval_retrieval \
--eval_data llm-embedder:qa/nq/train.json \
--corpus llm-embedder:qa/nq/corpus.json \
--metrics mrr recall collate_neg \
--save_name bge \
--data_root /data/llm-embedder

# BM25 (the result will be saved at llm-embedder:qa/nq/train.neg.bm25.json)
torchrun --nproc_per_node 8 -m evaluation.eval_retrieval \
--retrieval_method bm25 \
--eval_data llm-embedder:qa/nq/train.json \
--corpus llm-embedder:qa/nq/corpus.json \
--metrics mrr recall collate_neg \
--save_name bm25 \
--data_root /data/llm-embedder
```

## LM Scoring
Score positives and negatives in `eval_data` with $p(o|q,k)$ where $o$ is the desired output (i.e. `answers` field), $q$ is the query, and $k$ is a key (could be positive or negative).

```bash
torchrun --nproc_per_node=8 run_lm_score.py \
--eval_data llm-embedder:qa/msmarco/train.json \
--data_root /data/llm-embedder \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--save_name llama2-7b-chat
```
Results will be saved at `/data/llm-embedder/qa/msmarco/train.scored.llama2-7b-chat.json`


## Known Issues
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