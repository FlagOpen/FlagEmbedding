import io
import os
import fcntl
import pathlib
import zipfile
import argparse
import requests
import warnings
import itertools
from tqdm import tqdm
from datasets import load_dataset
from create.utils import save_tsv_dict, save_file_jsonl


# Load dataset
def download_source(source_dir):
    src = source_dir / "ds1000.py"
    url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000.py?raw=true"
    lock = src.with_suffix(".lock")
    with open(lock, "w") as f_lock:
        fcntl.flock(f_lock, fcntl.LOCK_EX)
        if not src.exists():
            warnings.warn(f"DS-1000 source is being saved to {src}.")
            print("Downloading source code...")
            r = requests.get(url, stream=True)
            with open(src, "wb") as f_src:
                f_src.write(r.content)
            open(src.parent / "__init__.py", "w").close()
            print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

def download_dataset(source_dir):
    path = source_dir / "ds1000_data"
    url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000_data.zip?raw=true"
    lock = path.with_suffix(".lock")
    with open(lock, "w") as f_lock:
        fcntl.flock(f_lock, fcntl.LOCK_EX)
        if not path.exists():
            warnings.warn(f"DS-1000 data is being saved to {path}.")
            print("Downloading dataset...")
            r = requests.get(url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(source_dir)
            print("Done.")
        fcntl.flock(f_lock, fcntl.LOCK_UN)

def get_dataset(source_dir, mode: str = "Completion", key: str = "All"):
    """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
    from ds.ds1000 import DS1000Dataset

    data = DS1000Dataset(source_dir / "ds1000_data", mode=mode).data
    if key == "All":
        if mode == "Insertion":
            warnings.warn(
                "Insertion not supported for Matplotlib. Only running others."
            )
            data = {k: v for k, v in data.items() if k != "Matplotlib"}
        dataset = list(itertools.chain(*data.values()))
    else:
        dataset = data[key]
    return dataset


# Collect queries, docs, and relations
def document2code(data: list):
    queries, docs, qrels = [], [], []

    # collect doc corpus
    code_docs = load_dataset("neulab/docprompting-conala", "docs")["train"]
    for i in range(len(code_docs)):
        docs.append({
            "_id": str(i),
            "title": code_docs[i]["doc_id"],
            "text": code_docs[i]["doc_content"],
            "metadata": {}
        })
    
    # load canonical docs
    ds1000 = load_dataset("json", data_files={"test": args.canonical_file})["test"]
    for idx,item in enumerate(tqdm(data)):
        example = item.data
        query = example["prompt"]
        query_id = f"{example['lib']}_{example['perturbation_origin_id']}"
        queries.append({"_id": query_id, "text": query, "metadata": {}})

        doc_ids = [doc["title"] for doc in ds1000[idx]["docs"]]
        for doc_id in doc_ids:
            corpus_id = code_docs["doc_id"].index(doc_id)
            corpus_id = str(corpus_id)
            qrels.append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})
    
    return queries, docs, qrels


def main():
    args.source_dir = pathlib.Path(__file__).parent.parent / args.source_dir
    os.makedirs(args.source_dir, exist_ok=True)
    download_source(args.source_dir)
    download_dataset(args.source_dir)
    dataset = get_dataset(args.source_dir, mode=args.mode, key=args.key)

    path = os.path.join(args.output_dir, f"ds1000_{args.key.lower()}_{args.mode.lower()}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    queries, docs, qrels = document2code(dataset)
    save_tsv_dict(qrels, os.path.join(path, "qrels", "test.tsv"), ["query-id", "corpus-id", "score"])
    
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="ds")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--mode", type=str, default="Completion", choices=["Completion", "Insertion"])
    parser.add_argument("--key", type=str, default="All", 
                        choices=["All", "Numpy", "Pandas", "Scipy", "Matplotlib", "Sklearn", "Tensorflow", "Pytorch"])
    parser.add_argument("--canonical_file", type=str, default="datasets/canonical/ds1000_docs.json")
    args = parser.parse_args()

    main()
