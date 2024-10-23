import os
import logging
import datasets
import requests
import gzip
import shutil
import tarfile
import json

from tqdm import tqdm, trange
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from huggingface_hub import snapshot_download
from FlagEmbedding.abc.evaluation.data_loader import AbsDataLoader

logger = logging.getLogger(__name__)

class MSMARCODataLoader(AbsDataLoader):
    def __init__(
        self,
        dataset_dir: str, # the dataset dir to load from
        cache_dir: str = None,
        token: str = None,
        dataset_name: str = 'msmarco',
        **kwargs
    ):
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.cache_dir = cache_dir
        self.token = token
        self.dataset_name = dataset_name
        self.sub_dataset_names = None
        self.split = 'test'
        if dataset_name == 'msmarco': self.split = 'dev'

        os.makedirs(self.dataset_dir, exist_ok=True)

        if dataset_name != 'cqadupstack':
            qrels_path = os.path.join(self.dataset_dir, 'qrels-{split}.json'.format(split=self.split))
            corpus_path = os.path.join(self.dataset_dir, 'corpus.json')
            queries_path = os.path.join(self.dataset_dir, 'queries-{split}.json'.format(split=self.split))
            queries, corpus, rels = {}, {}, {}
            if not os.path.exists(corpus_path):
                dataset = datasets.load_dataset(
                    'BeIR/{d}'.format(d=data_name),
                    'corpus',
                    trust_remote_code=True,
                    cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                    token=os.getenv('HF_TOKEN', token),
                    **kwargs
                )['corpus']
                for i in trange(len(dataset)):
                    corpus[str(dataset[i]['_id'])] = (dataset[i]['title'] + ' ' + dataset[i]['text']).strip()
                with open(corpus_path, 'w') as f:
                    json.dump(corpus, f)

            if not os.path.exists(queries_path):
                dataset = datasets.load_dataset(
                    'BeIR/{d}'.format(d=data_name), 
                    'queries', 
                    trust_remote_code=True,
                    cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                    token=os.getenv('HF_TOKEN', token),
                    **kwargs
                )['queries']
                for i in trange(len(dataset)):
                    queries[str(dataset[i]['_id'])] = dataset[i]['text'].strip()
                with open(queries_path, 'w') as f:
                    json.dump(queries, f)
            
            if not os.path.exists(qrels_path):
                dataset = datasets.load_dataset(
                    'BeIR/{d}-qrels'.format(d=data_name),
                    split=self.split, 
                    trust_remote_code=True,
                    cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                    token=os.getenv('HF_TOKEN', token),
                    **kwargs
                )
                for i in trange(len(dataset)):
                    if str(dataset[i]['query-id']) not in rels.keys():
                        rels[str(dataset[i]['query-id'])] = {}
                    rels[str(dataset[i]['query-id'])][str(dataset[i]['corpus-id'])] = int(dataset[i]['score'])
                with open(qrels_path, 'w') as f:
                    json.dump(rels, f)
        else:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
            data_path = util.download_and_unzip(url, dataset_dir)
            self.sub_dataset_names = list(os.listdir(data_path))
            for sub_dataset_name in self.sub_dataset_names:
                qrels_path = os.path.join(self.dataset_dir, sub_dataset_name, 'qrels-{split}.json'.format(split=self.split))
                corpus_path = os.path.join(self.dataset_dir, sub_dataset_name, 'corpus.json')
                queries_path = os.path.join(self.dataset_dir, sub_dataset_name, 'queries-{split}.json'.format(split=self.split))
                if not os.path.exists(corpus_path) or not os.path.exists(queries_path) or not os.path.exists(qrels_path):
                    full_path = os.path.join(data_path, sub_dataset_name)
                    corpus, queries, qrels = GenericDataLoader(data_folder=full_path).load(split="test")
                    for k in corpus_data.keys():
                        corpus[k] = {
                            (corpus[k]['title'] + ' ' + corpus[k]['text']).strip()
                        }

                    with open(corpus_path, 'w') as f:
                        json.dump(corpus, f)

                    with open(queries_path, 'w') as f:
                        json.dump(queries, f)

                    with open(qrels_path, 'w') as f:
                        json.dump(rels, f)


    def load_qrels(self):
        if self.sub_dataset_name is None:
            qrels_path = os.path.join(self.dataset_dir, 'qrels-{split}.json'.format(split=self.split))
            rels = json.load(open(qrels_path))
            return datasets.DatasetDict(rels)
        else:
            rels_list = []
            for sub_dataset_name in self.sub_dataset_names:
                qrels_path = os.path.join(self.dataset_dir, sub_dataset_name, 'qrels-{split}.json'.format(split=self.split))
                rels = json.load(open(qrels_path))
                rels_list.append(datasets.DatasetDict(rels))
            return rels_list


    def load_corpus(self):
        if self.sub_dataset_name is None:
            corpus_path = os.path.join(self.dataset_dir, 'corpus.json')
            corpus = json.load(open(corpus_path))
            for k in corpus.keys():
                corpus[k] = {
                    'text': corpus[k]
                }
            return datasets.DatasetDict(corpus)
        else:
            corpus_list = []
            for sub_dataset_name in self.sub_dataset_names:
                corpus_path = os.path.join(self.dataset_dir, sub_dataset_name, 'corpus.json')
                corpus = json.load(open(corpus_path))
                corpus_list.append(datasets.DatasetDict(corpus))
            return corpus_list

    def load_queries(self, split='dev'):
        if self.sub_dataset_name is None:
            queries_path = os.path.join(self.dataset_dir, 'queries-{split}.json'.format(split=self.split))
            qrels_path = os.path.join(self.dataset_dir, 'qrels-{split}.json'.format(split=self.split))
            queries = json.load(open(queries_path))
            rels = json.load(open(qrels_path))
            new_queries = {}
            for k in queries.keys():
                if k in rels.keys():
                    new_queries[k] = queries[k]
            return datasets.DatasetDict(new_queries)
        else:
            queries_list = []
            for sub_dataset_name in self.sub_dataset_names:
                queries_path = os.path.join(self.dataset_dir, sub_dataset_name, 'queries-{split}.json'.format(split=self.split))
                qrels_path = os.path.join(self.dataset_dir, sub_dataset_name, 'qrels-{split}.json'.format(split=self.split))
                queries = json.load(open(queries_path))
                rels = json.load(open(qrels_path))
                new_queries = {}
                for k in queries.keys():
                    if k in rels.keys():
                        new_queries[k] = queries[k]
                queries_list.append(datasets.DatasetDict(new_queries))
            return queries_list
