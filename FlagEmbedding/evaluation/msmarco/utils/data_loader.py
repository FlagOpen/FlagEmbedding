"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/data_loader.py
"""
import os
import logging
import datasets
import requests
import gzip
import shutil
import tarfile
import json

from tqdm import tqdm, trange
from huggingface_hub import snapshot_download
from FlagEmbedding.abc.evaluation.data_loader import AbsDataLoader

logger = logging.getLogger(__name__)

def download_file(
    url,
    directory,
    filename,
    file_type: str = 'dev'
):
    logger.warning(f"Trying to download dataset: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        source_file_path = os.path.join(directory, url.split('/')[-1])
        file_path = os.path.join(directory, filename)

        with open(source_file_path, 'wb') as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc="Downloading"):
                file.write(chunk)

        logger.warning(f"File downloaded: {source_file_path}")

        if url.endswith('.tar.gz'):
            extract_path = os.path.join(directory, 'extract_file')
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(source_file_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            
            results = {}
            with open(os.path.join(extract_path, 'queries.dev.tsv')) as f:
                for line in f:
                    if '\t' in line:
                        tmp = line.strip().split('\t')
                    else:
                        tmp = line.strip().split()
                    results[str(tmp[0])] = tmp[1]
            shutil.rmtree(extract_path)

        elif url.endswith('.gz'):
            
            temp_file_path = source_file_path.replace('.gz', '')

            with gzip.open(source_file_path, 'rb') as f_in:
                with open(temp_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(source_file_path)

            if temp_file_path.endswith('.jsonl'):
                results = {}
                with open(temp_file_path) as f:
                    for line in f:
                        tmp = json.loads(line)
                        results[str(list(tmp.keys())[0])] = list(tmp.values())[0]
            else:
                results = {}
                with open(temp_file_path) as f:
                    for line in f:
                        if '\t' in line:
                            tmp = line.strip().split('\t')
                        else:
                            tmp = line.strip().split()
                        if len(tmp) == 2:
                            results[str(tmp[0])] = tmp[1]
                        else:
                            if 'http' not in str(tmp[1]):
                                results[str(tmp[0])] = (tmp[2] + ' ' + tmp[3]).strip()
                            else:
                                if str(tmp[0]) not in results.keys():
                                    results[str(tmp[0])] = {}
                                results[str(tmp[0])][str(tmp[2])] = int(tmp[3])
            os.remove(temp_file_path)

        else:
            results = {}
            with open(source_file_path) as f:
                for line in f:
                    if '\t' in line:
                        tmp = line.strip().split('\t')
                    else:
                        tmp = line.strip().split()
                    if len(tmp) == 2:
                        results[str(tmp[0])] = tmp[1]
                    else:
                        if 'http' not in str(tmp[1]):
                            results[str(tmp[0])] = (tmp[2] + ' ' + tmp[3]).strip()
                        else:
                            if str(tmp[0]) not in results.keys():
                                results[str(tmp[0])] = {}
                            results[str(tmp[0])][str(tmp[2])] = int(tmp[3])
            os.remove(source_file_path)

        with open(file_path, 'w') as f:
            json.dump(results, f)

        logger.warning(f"File copied to: {file_path}")
        return file_path
    else:
        raise FileNotFoundError(f"Failed to download file: {url}. Status code: {response.status_code}")


class MSMARCODataLoader(AbsDataLoader):
    def __init__(
        self,
        dataset_dir: str, # the dataset dir to load from
        cache_dir: str = None,
        token: str = None,
        text_type: str = 'passage',
        **kwargs
    ):
        self.dataset_dir = os.path.join(dataset_dir, text_type)
        self.cache_dir = cache_dir
        self.token = token

        for split in ['dev', 'dl19', 'dl20']:
            corpus_path = os.path.join(self.dataset_dir, 'corpus.json')
            queries_path = os.path.join(self.dataset_dir, 'queries-{split}.json'.format(split=split))
            qrels_path = os.path.join(self.dataset_dir, 'qrels-{split}.json'.format(split=split))

            os.makedirs(self.dataset_dir, exist_ok=True)
            
            if text_type == 'passage':
                if not os.path.exists(corpus_path):
                    temp_dataset = datasets.load_dataset(
                        'Tevatron/msmarco-passage-corpus', 
                        'default', 
                        trust_remote_code=True,
                        cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                        token=os.getenv('HF_TOKEN', token),
                        **kwargs
                    )['train']
                    results = {}
                    for i in trange(len(temp_dataset), desc='load corpus'):
                        results[str(temp_dataset[i]['docid'])] = (temp_dataset[i]['title'] + ' ' + temp_dataset[i]['text']).strip()
                    
                    with open(corpus_path, 'w') as f:
                        json.dump(results, f)

                if split == 'dev':
                    if not os.path.exists(queries_path):
                        temp_dataset = datasets.load_dataset(
                            'BeIR/msmarco', 
                            'queries',
                            trust_remote_code=True,
                            cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                            token=os.getenv('HF_TOKEN', token),
                            **kwargs
                        )['queries']
                        
                        results = {}
                        for i in range(len(temp_dataset)):
                            results[str(temp_dataset[i]['_id'])] = (temp_dataset[i]['title'] + ' ' + temp_dataset[i]['text']).strip()
                        
                        with open(queries_path, 'w') as f:
                            json.dump(results, f)

                    if not os.path.exists(qrels_path):
                        temp_dataset = datasets.load_dataset(
                            'BeIR/msmarco-qrels', 
                            split='validation',
                            trust_remote_code=True,
                            cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                            token=os.getenv('HF_TOKEN', token),
                            **kwargs
                        )
                        results = {}
                        for i in range(len(temp_dataset)):
                            if str(temp_dataset[i]['query-id']) not in results.keys():
                                results[str(temp_dataset[i]['query-id'])] = {}
                            results[str(temp_dataset[i]['query-id'])][str(temp_dataset[i]['corpus-id'])] = int(temp_dataset[i]['score'])
                        
                        with open(qrels_path, 'w') as f:
                            json.dump(results, f)
                elif split == 'dl19':
                    if not os.path.exists(queries_path):
                        queries_path = download_file(
                            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
                            self.dataset_dir,
                            'queries-dl19.json'
                        )
                    if not os.path.exists(qrels_path):
                        qrels_path = download_file(
                            "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
                            self.dataset_dir,
                            'qrels-dl19.json'
                        )
                else:
                    if not os.path.exists(queries_path):
                        queries_path = download_file(
                            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
                            self.dataset_dir,
                            'queries-dl20.json'
                        )
                    if not os.path.exists(qrels_path):
                        qrels_path = download_file(
                            "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
                            self.dataset_dir,
                            'qrels-dl20.json'
                        )
            else:
                if not os.path.exists(corpus_path):
                    corpus_path = download_file(
                        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz",
                        self.dataset_dir,
                        'corpus.json'
                    )
                if split == 'dev':
                    if not os.path.exists(queries_path):
                        queries_path = download_file(
                            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
                            self.dataset_dir,
                            'queries-dev.json'
                        )
                    if not os.path.exists(qrels_path):
                        qrels_path = download_file(
                            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
                            self.dataset_dir,
                            'qrels-dev.json'
                        )
                elif split == 'dl19':
                    if not os.path.exists(queries_path):
                        queries_path = download_file(
                            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
                            self.dataset_dir,
                            'queries-dl19.json'
                        )
                    if not os.path.exists(qrels_path):
                        qrels_path = download_file(
                            "https://trec.nist.gov/data/deep/2019qrels-docs.txt",
                            self.dataset_dir,
                            'qrels-dl19.json'
                        )
                else:
                    if not os.path.exists(queries_path):
                        queries_path = download_file(
                            "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
                            self.dataset_dir,
                            'queries-dl20.json'
                        )
                    if not os.path.exists(qrels_path):
                        qrels_path = download_file(
                            "https://trec.nist.gov/data/deep/2020qrels-docs.txt",
                            self.dataset_dir,
                            'qrels-dl20.json'
                        )
    

    def load_qrels(self, split='dev'):
        qrels_path = os.path.join(self.dataset_dir, 'qrels-{split}.json'.format(split=split))
        rels = json.load(open(qrels_path))
        return datasets.DatasetDict(rels)

    def load_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, 'corpus.json')
        corpus = json.load(open(corpus_path))
        for k in corpus.keys():
            corpus[k] = {
                'text': corpus[k]
            }
        return datasets.DatasetDict(corpus)

    def load_queries(self, split='dev'):
        queries_path = os.path.join(self.dataset_dir, 'queries-{split}.json'.format(split=split))
        qrels_path = os.path.join(self.dataset_dir, 'qrels-{split}.json'.format(split=split))
        queries = json.load(open(queries_path))
        rels = json.load(open(qrels_path))
        new_queries = {}
        for k in queries.keys():
            if k in rels.keys():
                new_queries[k] = self.rels[k]
        return datasets.DatasetDict(new_queries)
