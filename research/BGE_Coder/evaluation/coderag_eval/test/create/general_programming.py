"""Aggregate all code-generation datasets."""

import os
import json
import datasets
import argparse
from create.utils import save_tsv_dict
from create.humaneval import document2code as d2c_humaneval
from create.mbpp import document2code as d2c_mbpp

D2C_FUNC_DICT = {
    "humaneval": d2c_humaneval,
    "mbpp": d2c_mbpp,
}
SPLIT_DICT = {
    "humaneval": ["test"], 
    "mbpp": ["train", "test", "validation", "prompt"],
}
HF_NAME_DICT = {
    "humaneval": "openai_humaneval",
    "mbpp": "mbpp",
}


def save_file_jsonl(data, path):
    with open(path,'w') as fw:
        for item in data:
            fw.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs='+', default=["humaneval", "mbpp"])
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--output_name", type=str, default="general-programming")
    args = parser.parse_args()

    path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(path)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    split_dict = {}
    for dataset_name in args.dataset_names:
        for split in SPLIT_DICT[dataset_name]:
            if split not in split_dict:
                split_dict[split] = []
            split_dict[split].append(dataset_name)
    
    dataset_dict = {
        dataset_name: datasets.load_dataset(HF_NAME_DICT[dataset_name])
        for dataset_name in args.dataset_names
    }
    docs, queries = [], []
    for split, ds_names in split_dict.items():
        for ds in ds_names:
            dataset = dataset_dict[ds]

            queries_split, docs_split, qrels_split = D2C_FUNC_DICT[ds](dataset, split)
            docs += docs_split
            queries += queries_split

        qrels_path = os.path.join(path, "qrels", f"{split}.tsv")
        save_tsv_dict(qrels_split, qrels_path, ["query-id", "corpus-id", "score"])
    
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))


if __name__ == "__main__":
    main()
