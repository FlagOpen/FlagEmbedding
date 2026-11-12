import argparse
import datasets
import os
from tqdm import tqdm
import random
from create.utils import save_tsv_dict, save_file_jsonl, load_jsonlines

def document2code(data, split="train"):
    data = data[split]
    code_search_net_data_queries = []
    code_search_net_data_docs = []
    code_search_net_data_qrels = []

    for item in tqdm(data):
        doc = item["func_documentation_string"]
        code = item["func_code_string"]
        doc_id = "{repository_name}_{func_path_in_repository}_{func_name}_doc".format_map(item)
        code_id = "{repository_name}_{func_path_in_repository}_{func_name}_code".format_map(item)
        code_search_net_data_queries.append({"_id": doc_id, "text": doc, "metadata": {}})
        code_search_net_data_docs.append({"_id": code_id, "title": item["func_name"], "text": code, "metadata": {}})
        code_search_net_data_qrels.append({"query-id": doc_id, "corpus-id": code_id, "score": 1})

    return code_search_net_data_queries, code_search_net_data_docs, code_search_net_data_qrels

def main():
#### /print debug information to stdout

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", help="codesearch net language")
    parser.add_argument("--output_dir", type=str, default="datasets")

    args = parser.parse_args()
    dataset = datasets.load_dataset("code_search_net", args.language)

    path = os.path.join(args.output_dir, "code_search_net_{}".format(args.language))
    os.makedirs(path)
    os.makedirs(os.path.join(path, "qrels"))

    docs = []
    queries = []
    for split in ["train", "validation", "test"]:
        queries_split, docs_split, qrels_split = document2code(dataset, split)
        docs += docs_split
        queries += queries_split

        save_tsv_dict(qrels_split, os.path.join(path, "qrels", "{}.tsv".format(split)), ["query-id", "corpus-id", "score"])

    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))

if __name__ == "__main__":
    main()
