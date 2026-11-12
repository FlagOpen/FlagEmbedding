import os
import argparse
import datasets
from tqdm import tqdm
from create.utils import save_tsv_dict, save_file_jsonl


def get_function_name(code: str) -> str:
    """Parse the function name for a code snippet string."""
    lines = code.split('\n')
    for line in lines:
        if line.lstrip().startswith("def "):
            break
    func_name = line.lstrip()[4: ]
    func_name = func_name.split('(')[0]
    return func_name


def document2code(data, split="test"):
    data = data[split]
    queries, docs, qrels = [], [], []

    for item in tqdm(data):
        doc = item["text"]
        code = "# " + item["text"] + '\n' + item["code"]
        doc_id = "{task_id}_doc".format_map(item)
        code_id = "{task_id}_code".format_map(item)

        queries.append({"_id": doc_id, "text": doc, "metadata": {}})
        docs.append({"_id": code_id, "title": get_function_name(item["code"]), "text": code, "metadata": {}})
        qrels.append({"query-id": doc_id, "corpus-id": code_id, "score": 1})
    
    return queries, docs, qrels


def main():
    dataset = datasets.load_dataset(args.dataset_name)

    path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    docs, queries = [], []
    for split in args.splits:
        queries_split, docs_split, qrels_split = document2code(dataset, split)
        docs += docs_split
        queries += queries_split

        qrels_path = os.path.join(path, "qrels", f"{split}.tsv")
        save_tsv_dict(qrels_split, qrels_path, ["query-id", "corpus-id", "score"])

        # create canonical file for test split if not existent yet
        if split == "test" and (not os.path.exists(args.canonical_file)):
            canonical_solutions = []
            for doc in docs_split:
                canonical_solutions.append([{
                    "text": doc["text"], "title": doc["title"]
                }])
            canonical_dataset = dataset["test"].add_column("docs", canonical_solutions)
            canonical_dataset.to_json(args.canonical_file)
    
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="google-research-datasets/mbpp")
    # parser.add_argument("--dataset_name", type=str, default="code-rag-bench/mbpp")
    parser.add_argument("--splits", type=str, default=["train", "validation", "test"],
                        choices=["train", "validation", "test", "prompt"])
    parser.add_argument("--output_name", type=str, default="mbpp")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--canonical_file", type=str, 
                        default="datasets/canonical/mbpp_solutions.json")
    
    args = parser.parse_args()
    main()
