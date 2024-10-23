"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/data_loader.py
"""
import os
import logging
import datasets
import requests
import gzip
import shutil

from tqdm import tqdm
from huggingface_hub import snapshot_download
from FlagEmbedding.abc.evaluation.data_loader import AbsDataLoader

logger = logging.getLogger(__name__)

def download_file(url, directory, filename):
    logger.warning(f"Trying to download dataset: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(directory, exist_ok=True)
        temp_file_path = os.path.join(directory, 'temp')
        file_path = os.path.join(directory, filename)
        with open(temp_file_path, 'wb') as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
                file.write(chunk)
        logger.warning(f"File downloaded: {temp_file_path}")
        # Check if the file is a .gz file
        if url.endswith('.gz'):
            with gzip.open(temp_file_path, 'rb') as f_in:
                with open(file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copyfile(temp_file_path, file_path)
        os.remove(file_path)
        logger.warning(f"File copied to: {file_path}")
        return file_path
    else:
        raise FileNotFoundError(f"Failed to download file: {url}. Status code: {response.status_code}")


class MSMARCOADataLoader(AbsDataLoader):
    def __init__(
        self,
        dataset_dir: str, # the dataset dir to load from
        cache_dir: str = None,
        token: str = None,
        text_type: str = 'passage',
        split: str = 'dev',
        **kwargs
    ):
        self.dataset_dir = dataset_dir
        self.query_dir = query_dir
        self.rels_dir = rels_dir
        self.cache_dir = cache_dir
        self.token = token

        self.corpus_path = os.path.join(self.dataset_dir, text_type, 'corpus.jsonl')
        self.queries_path = os.path.join(self.dataset_dir, text_type, 'queries-{split}.jsonl'.format(split=split))
        self.qrels_path = os.path.join(self.dataset_dir, text_type, 'qrels-{split}.tsv'.format(split=split))
        
        if text_type == 'passage':
            # if not os.path.exists(self.corpus_path):
            #     self.corpus_path = download_file(
            #         os.path.join(os.getenv("HF_ENDPOINT", "https://huggingface.co"), "datasets/Tevatron/msmarco-passage-corpus/resolve/main/corpus.jsonl.gz"),
            #         self.dataset_dir,
            #         os.path.join(text_type, 'corpus.jsonl')
            #     )
            if split == 'dev':
                if not os.path.exists(self.queries_path):
                    self.queries_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'queries-dev.jsonl')
                    )
                if not os.path.exists(self.qrels_path):
                    self.qrels_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
                        self.dataset_dir,
                        os.path.join(text_type, 'qrels-dev.tsv')
                    )
            elif split == 'dl19':
                if not os.path.exists(self.queries_path):
                    self.queries_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'queries-dev.jsonl')
                    )
                if not os.path.exists(self.qrels_path):
                    self.qrels_path = download_file(
                        "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
                        self.dataset_dir,
                        os.path.join(text_type, 'qrels-dev.tsv')
                    )
            else:
                if not os.path.exists(self.queries_path):
                    self.queries_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'queries-dev.jsonl')
                    )
                if not os.path.exists(self.qrels_path):
                    self.qrels_path = download_file(
                        "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
                        self.dataset_dir,
                        os.path.join(text_type, 'qrels-dev.tsv')
                    )
        else:
            if not os.path.exists(self.corpus_path):
                self.corpus_path = download_file(
                    "https://msmarco.z22.web.core.windows.net/msmarcoranking/fulldocs.tsv.gz",
                    self.dataset_dir,
                    os.path.join(text_type, 'corpus.jsonl')
                )
            if split == 'dev':
                if not os.path.exists(self.queries_path):
                    self.queries_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'queries-dev.jsonl')
                    )
                if not os.path.exists(self.qrels_path):
                    self.qrels_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'qrels-dev.tsv')
                    )
            elif split == 'dl19':
                if not os.path.exists(self.queries_path):
                    self.queries_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'queries-dev.jsonl')
                    )
                if not os.path.exists(self.qrels_path):
                    self.qrels_path = download_file(
                        "https://trec.nist.gov/data/deep/2019qrels-docs.txt",
                        self.dataset_dir,
                        os.path.join(text_type, 'qrels-dev.tsv')
                    )
            else:
                if not os.path.exists(self.queries_path):
                    self.queries_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
                        self.dataset_dir,
                        os.path.join(text_type, 'queries-dev.jsonl')
                    )
                if not os.path.exists(self.qrels_path):
                    self.qrels_path = download_file(
                        "https://trec.nist.gov/data/deep/2020qrels-docs.txt",
                        self.dataset_dir,
                        os.path.join(text_type, 'qrels-dev.tsv')
                    )
    

    def load_qrels(self, split: str = 'test'):
        qrels_path = os.path.join(self.dataset_dir, f"{split}_qrels.jsonl")
        qrels_data = datasets.load_dataset('json', data_files=qrels_path, cache_dir=self.cache_dir)['train']

        qrels = {}
        for data in qrels_data:
            qid = data['qid']
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][data['docid']] = data['relevance']

        return datasets.DatasetDict(qrels)

    def load_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, 'corpus.jsonl')
        corpus_data = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']
        
        corpus = {e['id']: {'text': e['text']} for e in corpus_data}
        return datasets.DatasetDict(corpus)

    def load_queries(self, split: str = 'test'):
        queries_path = os.path.join(self.dataset_dir, f"{split}_queries.jsonl")
        queries_data = datasets.load_dataset('json', data_files=queries_path, cache_dir=self.cache_dir)['train']

        queries = {e['id']: e['text'] for e in queries_data}
        return datasets.DatasetDict(queries)
