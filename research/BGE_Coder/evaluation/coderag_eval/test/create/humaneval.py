import os
import argparse
import datasets
from tqdm import tqdm
from create.utils import save_tsv_dict, save_file_jsonl


def document2code(data, split="test"):
    data = data[split]
    queries, docs, qrels = [], [], []

    for item in tqdm(data):
        doc = item["prompt"]
        code = item["prompt"] + '\n' + item["canonical_solution"]
        doc_id = "{task_id}_doc".format_map(item)
        code_id = "{task_id}_code".format_map(item)

        queries.append({"_id": doc_id, "text": doc, "metadata": {}})
        docs.append({"_id": code_id, "title": item["entry_point"], "text": code, "metadata": {}})
        qrels.append({"query-id": doc_id, "corpus-id": code_id, "score": 1})
    
    return queries, docs, qrels


def main():
    dataset = datasets.load_dataset(args.dataset_name)

    path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    queries, docs, qrels = document2code(dataset, split="test")
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
    qrels_path = os.path.join(path, "qrels", "test.tsv")
    save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])

    # create canonical file if not existent yet
    if not os.path.exists(args.canonical_file):
        canonical_solutions = []
        for doc in docs:
            canonical_solutions.append([{
                "text": doc["text"], "title": doc["title"]
            }])
        canonical_dataset = dataset["test"].add_column("docs", canonical_solutions)
        canonical_dataset.to_json(args.canonical_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="openai_humaneval")
    parser.add_argument("--output_name", type=str, default="humaneval")
    parser.add_argument("--canonical_file", type=str, 
                        default="datasets/canonical/humaneval_solutions.json")
    parser.add_argument("--output_dir", type=str, default="datasets")
    args = parser.parse_args()

    main()
