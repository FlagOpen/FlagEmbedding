import io
import os
import glob
import json
import argparse
import requests
import zipfile
from collections import defaultdict
from create.utils import save_tsv_dict, save_file_jsonl

REPOs_line_and_api = [
    'huggingface_diffusers',
    'nerfstudio-project_nerfstudio',
    'awslabs_fortuna',
    'huggingface_evaluate',
    'google_vizier',
    'alibaba_FederatedScope',
    'pytorch_rl',
    'opendilab_ACE',
]

REPOs_function = [
    "amazon-science_patchcore-inspection",
    "deepmind_tracr",
    "facebookresearch_omnivore",
    "google_lightweight_mmm",
    "lucidrains_imagen-pytorch",
    "maxhumber_redframes",
]

REPO_DIRs = {
    "api": "repositories/line_and_api_level",
    "line": "repositories/line_and_api_level",
    "function": "repositories/function_level",
}


def iterate_repository(base_dir: str, repo: str) -> dict:
    pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
    files = glob.glob(pattern, recursive=True)

    skipped_files = []
    loaded_code_files = dict()
    base_dir_list = os.path.normpath(base_dir).split(os.sep)
    for fname in files:
        try:
            code = open(fname, 'r', encoding='utf8').read()
            fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
            loaded_code_files[fpath_tuple]= code
        except Exception as e:
            skipped_files.append((fname, e))
            continue

    if len(skipped_files) > 0:
        print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
        for fname, e in skipped_files:
            print(f"{fname}: {e}")
    return loaded_code_files


def window_overlap(span: tuple, target_span: tuple) -> bool:
    if span[0] >= target_span[1] or span[1] <= target_span[0]:
        return False
    return True


class RepoWindowMaker:
    def __init__(self, base_dir, repo, tasks, window_size, slice_size):
        self.base_dir = base_dir
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = 1 if window_size // slice_size == 0 else window_size // slice_size
        self.tasks = tasks
        self.source_code_files = iterate_repository(base_dir, repo)
        
    def _buid_windows_for_a_file(self, fpath_tuple, code):
        code_windows = []
        code_lines = code.splitlines()
        delta_size = self.window_size // 2
        for line_no in range(0, len(code_lines), self.slice_step): # line_no starts from 0
            start_line_no = max(0, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            if not window_lines:  # all empty lines
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'repo': self.repo,
                    'slice_size': self.slice_size,
                }
            })
        return code_windows
    
    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines

    def build_windows(self):
        all_code_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            all_code_windows += self._buid_windows_for_a_file(fpath_tuple, code)
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        print(f'build {len(merged_code_windows)} windows for {self.repo} with window size {self.window_size} and slice {self.slice_size}')
        ground_truth_indices = {}
        for task in self.tasks:
            fpath_tuple = tuple(task['metadata']['fpath_tuple'])
            line_no = task['metadata']['line_no']
            start_line_no = task['metadata']['context_start_lineno']
            for i, window in enumerate(merged_code_windows):
                if window["metadata"][0]["fpath_tuple"] != fpath_tuple:
                    continue
                if any([
                    window_overlap(
                        (sub_window["start_line_no"], sub_window["end_line_no"]), 
                        (start_line_no, line_no + 1)
                    )
                    for sub_window in window["metadata"]
                ]):
                    if i not in ground_truth_indices: 
                        ground_truth_indices[i] = []
                    ground_truth_indices[i].append(task["metadata"]["task_id"])
                    
        return merged_code_windows, ground_truth_indices


def download_data(directory: str = "repoeval"):
    os.makedirs(directory, exist_ok=True)
    
    datasets_dir = os.path.join(directory, "datasets")
    repos_lineapi_dir = os.path.join(directory, "repositories", "line_and_api_level")
    repos_function_dir = os.path.join(directory, "repositories", "function_level")

    print(f"Start downloading the necessary `datasets` and `repositories` files.")
    if not os.path.exists(datasets_dir):
        print(f"Start downloading the `datasets`.")
        datasets_url = "https://github.com/microsoft/CodeT/raw/main/RepoCoder/datasets/datasets.zip"
        r = requests.get(datasets_url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(datasets_dir)
        print("Finished downloading the `datasets` files.")

    if not os.path.exists(repos_lineapi_dir):
        print(f"Start downloading the `repositories` (line_and_api).")
        repos_lineapi_url = "https://github.com/microsoft/CodeT/raw/main/RepoCoder/repositories/line_and_api_level.zip"
        r = requests.get(repos_lineapi_url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(repos_lineapi_dir)
    
    if not os.path.exists(repos_function_dir):
        print(f"Start downloading the `repositories` (function).")
        # repos_function_url = "https://github.com/microsoft/CodeT/raw/main/RepoCoder/repositories/function_level.zip"
        repos_function_url = "https://github.com/Veronicium/repoeval_debug/raw/main/function_level.zip"
        r = requests.get(repos_function_url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(repos_function_dir)
        print("Finished downloading the `repositories` files.")


def repo2code(
    repo: str, data_cache_dir: str, 
    split: str, context_length: str,
    window_size: int, slice_size: int
):
    # load test examples
    file_name = f"{split}_level_completion_{context_length}_context_codex.test.jsonl"
    if split == 'function':
        file_name = file_name.replace('.test.jsonl', '.test.clean.jsonl')
    
    task_path = os.path.join(data_cache_dir, "datasets", file_name)
    tasks = [json.loads(l.rstrip()) for l in open(task_path, 'r')]
    tasks = [task for task in tasks if repo == task['metadata']['task_id'].split('/')[0]]

    # collect queries
    queries = []
    for task in tasks:
        query_id = task["metadata"]["task_id"]
        # text = '\n'.join(task["prompt"].split('\n')[-2:])
        text = task["prompt"]
        metadata = task["metadata"]
        queries.append({"_id": query_id, "text": text, "metadata": metadata})
    
    base_dir = os.path.join(data_cache_dir, REPO_DIRs[split])
    repo_window_maker = RepoWindowMaker(base_dir, repo, tasks, window_size, slice_size)
    windows, ground_truth_indices = repo_window_maker.build_windows()
    corpus, qrels = [], []
    query_id2gt = {task['metadata']['task_id']:[] for task in tasks}
    for i, window in enumerate(windows):
        path = '-'.join(window["metadata"][0]["fpath_tuple"])
        line = f"{window['metadata'][0]['start_line_no']}-{window['metadata'][-1]['end_line_no']}"
        corpus_id = f"{repo}_{path}_{line}"
        corpus.append({
            "_id": corpus_id, "title": path, 
            "text": window["context"], "metadata": window["metadata"]
        })
        if i in ground_truth_indices:
            for query_id in ground_truth_indices[i]:
                qrels.append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})
                query_id2gt[query_id].append({"title": corpus_id.replace('_', '/'), "text": window["context"]})

    return queries, corpus, qrels, query_id2gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--split", type=str, required=True, choices=["api", "line", "function"])
    parser.add_argument("--context_length", type=str, default="1k", choices=["1k", "2k", "4k"])
    parser.add_argument("--data_cache_dir", type=str, default="output/repoeval")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--slice_size", type=int, default=2)
    args = parser.parse_args()

    download_data(args.data_cache_dir)

    path = os.path.join(args.output_dir, "repoeval", args.split)
    os.makedirs(path, exist_ok=True)
    REPOs = REPOs_function if args.split == "function" else REPOs_line_and_api
    
    file_name = f"{args.split}_level_completion_{args.context_length}_context_codex.test.jsonl"
    data_path = os.path.join(args.data_cache_dir, "datasets", file_name)
    data = [json.loads(l.rstrip()) for l in open(data_path, 'r')]
    
    # preprocess function completion data (the data in the RepoCoder repo isn't correctly formatted)
    if args.split == 'function':        
        repo2idx = {}
        for task in data:
            repo = task['metadata']['task_id'].replace('--', '_').split('/')[0]
            if repo not in repo2idx:
                repo2idx[repo] = 0
            task['metadata']['task_id'] = task['metadata']['task_id'].replace('--', '_').replace('idx', str(repo2idx[repo]))
            task['metadata']['line_no'] = task['metadata']['lineno']
            repo2idx[repo] += 1
            
        new_data_path = data_path.replace('.test.jsonl', '.test.clean.jsonl')
        with open(new_data_path, 'w') as f:
            for task in data:
                repo = task['metadata']['task_id'].split('/')[0]
                if repo not in REPOs:
                    continue
                f.write(json.dumps(task) + '\n')
                
        data = [json.loads(l.rstrip()) for l in open(new_data_path, 'r')]

    # build query, docs, and qrels for each repository
    queries, corpus, qrels = [], [], []
    query_id2gt = {}
    for repo in REPOs:
        repo_queries, repo_corpus, repo_qrels, repo_query_id2gt = repo2code(
            repo, args.data_cache_dir, 
            args.split, args.context_length,
            args.window_size, args.slice_size
        )
        queries += repo_queries
        corpus += repo_corpus
        qrels += repo_qrels
        query_id2gt.update(repo_query_id2gt)

    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(corpus, os.path.join(path, "corpus.jsonl"))
    save_tsv_dict(qrels, os.path.join(path, "qrels", "test.tsv"), ["query-id", "corpus-id", "score"])
    
    gt_data = []
    for example in data:
        query_id = example['metadata']['task_id']
        gt = query_id2gt[query_id]
        new_example = {
            "prompt": example["prompt"],
            "reference": example["metadata"]["ground_truth"],
            "docs": gt[:10],
            "metadata": {k:v for k,v in example["metadata"].items() if k != "ground_truth"},
        }
        gt_data.append(new_example)
        
    results_file = os.path.join(args.results_dir, f"repoeval-{args.split}-{args.context_length}-gt.jsonl")
    with open(results_file, "w") as fw:
        for ex in gt_data:
            fw.write(json.dumps(ex) + "\n")
        
    results_file = os.path.join(args.results_dir, f"repoeval-{args.split}-{args.context_length}-infile.jsonl")
    with open(results_file, "w") as fw:
        for ex in gt_data:
            ex = {k:v for k,v in ex.items() if k != "docs"}
            ex["docs"] = []
            fw.write(json.dumps(ex) + "\n")

if __name__ == "__main__":
    main()
