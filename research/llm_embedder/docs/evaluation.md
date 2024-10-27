# Evaluation

LLM-Embedder supports 6 retrieval-augmentation tasks tailored for modern LLMs, including:
- Question Answering (qa)
  - evaluate with `eval_popqa` and `eval_mmlu`
- In-Context Learning (icl)
  - evaluate with `eval_icl`
- Long Conversation (chat)
  - evaluate with `eval_msc`
- Long-Range Language Modeling (lrlm)
  - evaluate with `eval_lrlm`
- Tool Learning (tool)
  - evaluate with `eval_tool`
- Conversational Search (convsearch)
  - evaluate with `eval_qrecc`

## Environment
It is recommended that you create a new environment:
```
cd FlagEmbedding/llm_embedder

conda env create -f environment.yaml --name llm-embedder
conda activate llm-embedder
```

To use BM25, you must download **java11** and **anserini**, then add java to your `PATH`:
```bash
# feel free to alternate /data to your prefered location
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/java11.tar.gz?download=true -O /data/java11.tar.gz
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/anserini.tar.gz?download=true -O /data/anserini.tar.gz

cd /data
tar -xzvf java11.tar.gz
tar -xzvf anserini.tar.gz

# below just temporarily set JAVA_HOME; it is RECOMMENDED that you store the lines the setting in ~/.bashrc
export JAVA_HOME=/data/jdk-11.0.2
export PATH=$JAVA_HOME/bin:$PATH
```

## Data
You should download the data for fine-tuning & evaluation then untar the file at anywhere you prefer, e.g. `/data`, which results in a folder `/data/llm-embedder`:
```bash
# feel free to alternate /data to your prefered location
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/llm-embedder.tar.gz?download=true -O /data/llm-embedder.tar.gz

cd /data
tar -xzvf llm-embedder-eval.tar.gz
```

The corpus of QReCC for conversational search is too large (54M passages), we separately upload it to huggingface datasets [namespace-Pt/qrecc-corpus](https://huggingface.co/datasets/namespace-Pt/qrecc-corpus). To evaluate the performance on conversational search, you should load it and save it as json file in the `qrecc` folder:
```python
import datasets
# load dataset
qrecc_corpus = datasets.load_dataset("namespace-Pt/qrecc-corpus", split="train")
# save to jsonline format in YOUR data folder
qrecc_corpus.to_json("/data/llm-embedder/convsearch/qrecc/corpus.json", force_ascii=False, lines=True, orient="records")
```

## Benchmark
### Commands
Below are commands to run evaluation for different retrieval models. You can replace `eval_popqa` with any of `eval_mmlu`, `eval_icl`, `eval_lrlm`, `eval_msc`, `eval_tool`, and `eval_qrecc`. The results will be logged at `data/results/`. 

*All our evaluation are based on `meta-llama/Llama-2-7b-chat-hf`. To use different language models, e.g. `Qwen/Qwen-7B-Chat`, simply add `--model_name_or_path Qwen/Qwen-7B-Chat` after every command.*

*Note that you can modify the default value of `data_root` in `src/retrieval/args.py`, so that you don't need to type it for each command.*

```bash
cd FlagEmbedding/llm_embedder

# No retrieval
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --retrieval_method no --data_root /data/llm-embedder

# Random
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --retrieval_method random --data_root /data/llm-embedder

# BM25 (anserini_dir is the folder where you untar anserini.tar.gz)
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --retrieval_method bm25 --data_root /data/llm-embedder --anserini_dir /data/anserini

# Contriever
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder facebook/Contriever --dense_metric ip --add_instruction False --data_root /data/llm-embedder

# BGE
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder BAAI/bge-base-en --version bge --data_root /data/llm-embedder

# AAR (uses special decoder pooling)
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder OpenMatch/AAR-ANCE --pooling_method decoder --add_instruction False --data_root /data/llm-embedder

# APIRetriever
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder ToolBench/ToolBench_IR_bert_based_uncased --pooling_method mean --dense_metric ip --add_instruction False --data_root /data/llm-embedder

# LLMRetriever
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder intfloat/llm-retriever-base --add_instruction false --pooling_method mean --data_root /data/llm-embedder

# RetroMAE_BEIR
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder Shitao/RetroMAE_BEIR --dense_metric ip --add_instruction False --data_root /data/llm-embedder

# LLM Embedder
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder BAAI/llm-embedder --version llm-embedder --data_root /data/llm-embedder
```

For Instructor, we should first convert it to our format:
```python
# convert sentence transformer based Instructor to our format
import torch
from src.retrieval import DenseRetriever, RetrievalArgs
from sentence_transformers import SentenceTransformer

model_args = RetrievalArgs(
    query_encoder = "hkunlp/instructor-base",
    pooling_method = ["mean", "dense"],
    dtype = "fp32"
)
retriever = DenseRetriever(**asdict(model_args), cache_dir=model_args.model_cache_dir)
tokenizer = retriever.tokenizer

with torch.no_grad():
    sent_model = SentenceTransformer(model_args.query_encoder, device="cpu")
    retriever.dense_pooler.weight.data = sent_model.state_dict()["2.linear.weight"]

    x = sent_model.encode(["I love you"])
    y = retriever.encode("I love you")
    print(torch.isclose(torch.from_numpy(x), y))
    retriever.save_pretrained("data/outputs/instructor-base")
```
Then we evaluate with 
```bash
torchrun --nproc_per_node 8 -m evaluation.eval_popqa --query_encoder data/outputs/instructor-base/encoder --pooling_method mean dense --version instructor --data_root /data/llm-embedder
```


### Leaderboard
All the following results are based on `meta-llama/Llama-27b-chat-hf` with `torch==2.0.1`, `transformers==4.30.0` on a `8xA100` machine with `CUDA==11.4`.

|Model|MMLU (avg)|PopQA (acc)|In-Context Learning (avg)|Long Conversation (ppl)|Long-Range Language Modeling (ppl)|Tool Learning (ndcg)|Conversational Search (ndcg)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|None|0.4599|0.2061|0.4645|19.3501|6.4003|--|--|
|BM25|0.4721|0.3491|0.484|14.6512|6.1558|0.5115|0.4341|
|Instructor|0.4721|0.3533|0.6036|14.8799|6.1733|0.3882|0.2863|
|Contriever|0.4684|0.3276|0.6009|14.2129|6.1305|0.4904|0.3563|
|BGE|0.4896|0.4491|0.5974|14.2943|6.1335|0.5761|0.3856|
|AAR|0.4826|0.4792|0.5938|14.6999|6.1528|0.42|0.2877|
|LLMRetriever|0.4625|0.2506|0.6262|14.4746|6.1750|0.1321|0.0234|
|APIRetriever|0.4625|0.2488|0.5945|14.7834|6.1833|0.8017|0.1137|
|LLM-Embedder (ours)|**0.4903**|**0.5052**|**0.6288**|**13.4832**|**6.0972**|**0.8645**|**0.5053**|

