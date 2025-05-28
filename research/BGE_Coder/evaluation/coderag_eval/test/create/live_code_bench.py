import os
import argparse
import datasets
from tqdm import tqdm
from datasets import load_dataset
from create.utils import save_tsv_dict, save_file_jsonl


def get_queries(data, split="test") -> list[dict]:
    queries = [{
        "_id": item["question_id"] + '__' + item["contest_id"], 
        "text": item["question_content"], 
        "metadata": {}
    } for item in data[split]]
    return queries

def get_corpus(hf_name: str, cache_dir: str) -> list[dict]:
    dataset = load_dataset(hf_name, cache_dir=cache_dir)["train"]
    corpus = [
        {"_id": i, "text": item["text"], "title": item["title"]}
        for i,item in enumerate(dataset)
    ]
    return corpus


def main():
    dataset = datasets.load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    queries = get_queries(dataset, split="test")
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))

    docs = get_corpus(args.corpus_name, args.cache_dir)
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))

    qrels = []  # no ground-truth solutions
    qrels_path = os.path.join(path, "qrels", "test.tsv")
    save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="livecodebench/code_generation")
    parser.add_argument("--corpus_name", type=str, default="code-rag-bench/programming-solutions")
    parser.add_argument("--cache_dir", type=str, default="/scratch/zhiruow/data")
    parser.add_argument("--output_name", type=str, default="livecodebench")
    parser.add_argument("--output_dir", type=str, default="datasets")
    args = parser.parse_args()

    main()
