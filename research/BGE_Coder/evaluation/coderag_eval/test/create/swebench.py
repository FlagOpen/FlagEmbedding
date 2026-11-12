import os
import re
import chardet
import unidiff
import argparse
import datasets
import traceback
import subprocess
from git import Repo
from tqdm import tqdm
from pathlib import Path
from tempfile import TemporaryDirectory
from create.utils import save_tsv_dict, save_file_jsonl

# %% Get oracle file contents

# get oracle file contents from the repo
class ContextManager:
    def __init__(self, repo_path, base_commit, verbose=False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.old_dir = os.getcwd()
        self.base_commit = base_commit
        self.verbose = verbose

    def __enter__(self):
        os.chdir(self.repo_path)
        cmd = f"git reset --hard {self.base_commit} && git clean -fdxq"
        if self.verbose:
            subprocess.run(cmd, shell=True, check=True)
        else:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return self

    def get_environment(self):
        raise NotImplementedError()  # TODO: activate conda environment and return the environment file

    def get_readme_files(self):
        files = os.listdir(self.repo_path)
        files = list(filter(lambda x: os.path.isfile(x), files))
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.old_dir)


class AutoContextManager(ContextManager):
    """Automatically clones the repo if it doesn't exist"""

    def __init__(self, instance, root_dir=None, verbose=False, token=None):
        if token is None:
            token = os.environ.get("GITHUB_TOKEN", "git")
        self.tempdir = None
        if root_dir is None:
            self.tempdir = TemporaryDirectory()
            root_dir = self.tempdir.name
        self.root_dir = root_dir
        repo_dir = os.path.join(self.root_dir, instance["repo"].replace("/", "__"))
        if not os.path.exists(repo_dir):
            repo_url = (
                f"https://{token}@github.com/swe-bench/"
                + instance["repo"].replace("/", "__")
                + ".git"
            )
            if verbose:
                print(f"Cloning {instance['repo']} to {root_dir}")
            Repo.clone_from(repo_url, repo_dir)
        super().__init__(repo_dir, instance["base_commit"], verbose=verbose)
        self.instance = instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tempdir is not None:
            self.tempdir.cleanup()
        return super().__exit__(exc_type, exc_val, exc_tb)


def ingest_files(filenames):
    files_dict = dict()
    for filename in filenames:
        with open(filename) as f:
            content = f.read()
        files_dict[filename] = content
    return files_dict

def get_oracle_filenames(instance):
    """
    Returns the filenames that are changed in the patch
    """
    source_files = {
        patch_file.source_file.split("a/", 1)[-1]
        for patch_file in unidiff.PatchSet(instance["patch"])
    }
    gold_docs = set()
    for source_file in source_files:
        gold_docs.add(source_file)
    return gold_docs


# get all file contents from the repo
def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", name.lower()))
    return any(word in words for word in test_phrases)

def list_files(root_dir, include_tests=False):
    files = []
    for filename in Path(root_dir).rglob("*.py"):
        if not include_tests and is_test(filename.as_posix()):
            continue
        files.append(filename.relative_to(root_dir).as_posix())
    return files

def detect_encoding(filename):
    """
    Detect the encoding of a file
    """
    with open(filename, "rb") as file:
        rawdata = file.read()
    return chardet.detect(rawdata)["encoding"]

def ingest_directory_contents(root_dir, include_tests=False):
    files_content = {}
    for relative_path in list_files(root_dir, include_tests=include_tests):
        filename = os.path.join(root_dir, relative_path)
        encoding = detect_encoding(filename)
        if encoding is None:
            content = "[BINARY DATA FILE]"
        else:
            try:
                with open(filename, encoding=encoding) as file:
                    content = file.read()
            except (UnicodeDecodeError, LookupError):
                content = "[BINARY DATA FILE]"
        files_content[relative_path] = content
    return files_content

def get_file_contents(input_instances, verbose: bool = False, tmp_dir: str = "/scratch"):
    orig_dir = os.getcwd()
    with TemporaryDirectory(dir=tmp_dir if os.path.exists(tmp_dir) else "/tmp") as root_dir:
        for instance_id, instance in tqdm(
            input_instances.items(),
            total=len(input_instances),
            desc="Getting file contents",
        ):
            try:
                with AutoContextManager(instance, root_dir, verbose=verbose) as cm:
                    readmes = cm.get_readme_files()
                    instance["readmes"] = ingest_files(readmes)
                    instance["oracle_file_contents"] = ingest_files(get_oracle_filenames(instance))
                    instance["file_contents"] = ingest_directory_contents(cm.repo_path)
                    assert all([
                        okey in instance["file_contents"] 
                        for okey in instance["oracle_file_contents"].keys()
                    ])
            except Exception as e:
                print(f"Failed on instance {instance_id}", e)
                traceback.print_exc()
            finally:
                # if AutoContextManager fails to exit properly future exits will return the wrong directory
                os.chdir(orig_dir)
    os.chdir(orig_dir)


# %% Get queries, docs, and qrels

def document2code(data, split: str = "test"):
    subset = data[split]
    if args.num_examples is not None:
        import random
        indices = random.sample([i for i in range(len(subset))], args.num_examples)
        subset = subset.select(indices)
    print(subset)

    # get queries for each example
    queries = [
        {
            "_id": item["instance_id"],
            "text": item["problem_statement"], 
            "metadata": {}
        }
        for item in subset
    ]

    subset_dict = {x["instance_id"]: x for x in subset}
    get_file_contents(subset_dict, tmp_dir=args.tmp_dir)

    # collect all docs, i.e., code chunks from the repo
    docs = []
    for instance_id, instance in subset_dict.items():
        print(f"Instance #{instance_id}: {len(instance['oracle_file_contents'])} oracle / {len(instance['file_contents'])} files")
        for filename, content in instance["file_contents"].items():
            docs.append({
                "_id": f"{instance_id}_{filename}",
                "title": filename,
                "text": content,
                "metadata": {},
            })

    # find ground-truth docs for each example
    qrels = []
    for instance_id, instance in subset_dict.items():
        for filename, content in instance["oracle_file_contents"].items():
            qrels.append({
                "query-id": instance_id,
                "corpus-id": f"{instance_id}_{filename}",
                "score": 1
            })
    
    return queries, docs, qrels


def main():
    dataset = datasets.load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    name = "swe-bench"
    if "lite" in args.dataset_name.lower():
        name += "-lite"
        
    path = os.path.join(args.output_dir, name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    queries, docs, qrels = document2code(dataset, split="test")
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
    qrels_path = os.path.join(path, "qrels", "test.tsv")
    save_tsv_dict(qrels, qrels_path, ["query-id", "corpus-id", "score"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Lite",
                        choices=["princeton-nlp/SWE-bench", "princeton-nlp/SWE-bench_Lite"])
    parser.add_argument("--cache_dir", type=str, default="/scratch/zhiruow/data")
    parser.add_argument("--tmp_dir", type=str, default="/scratch/zhiruow/tmp")
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    main()
