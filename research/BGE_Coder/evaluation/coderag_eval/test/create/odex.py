import os
import re
import random
import argparse
import datasets
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from create.utils import save_tsv_dict, save_file_jsonl


def document2code(data, split="test"):
    data = data[split]
    queries, docs, qrels = [], [], []

    # build doc corpus
    code_docs = load_dataset("neulab/docprompting-conala", "docs")["train"]
    for i in range(len(code_docs)):
        docs.append({
            "_id": str(i),
            "title": code_docs[i]["doc_id"],
            "text": code_docs[i]["doc_content"],
            "metadata": {}
        })
    
    # load canonical docs
    odex = load_dataset("json", data_files={"test": args.canonical_file})["test"]
    # collect queries and query-doc matching
    for idx,item in enumerate(tqdm(data)):
        query = item["intent"]
        query_id = f"{idx}_{item['task_id']}"
        queries.append({"_id": query_id, "text": query, "metadata": {}})

        doc_ids = [doc["title"] for doc in odex[idx]["docs"]]
        for doc_id in doc_ids:
            corpus_id = code_docs["doc_id"].index(doc_id)
            corpus_id = str(corpus_id)
            qrels.append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})
    
    return queries, docs, qrels


def main():
    if '_' in args.dataset_name:
        dataset_name = args.dataset_name.split('_')[0]
        language = args.dataset_name.split('_')[1]
    else:
        dataset_name = args.dataset_name
        language = 'en'
    dataset = datasets.load_dataset(dataset_name, language) # english version by default

    path = os.path.join(args.output_dir, args.output_name.replace('en', language))
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    docs, queries = [], []
    for split in ["test"]:
        queries_split, docs_split, qrels_split = document2code(dataset, split)
        docs += docs_split
        queries += queries_split

        save_tsv_dict(qrels_split, os.path.join(path, "qrels", "{}.tsv".format(split)), ["query-id", "corpus-id", "score"])
    
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="neulab/odex")
    parser.add_argument("--output_name", type=str, default="odex_en")
    parser.add_argument("--canonical_file", type=str, default="datasets/canonical/odex_docs.json")
    parser.add_argument("--output_dir", type=str, default="datasets")
    args = parser.parse_args()

    main()
