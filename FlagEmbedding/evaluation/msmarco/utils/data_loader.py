"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/data_loader.py
"""
import os
import logging
import datasets

from huggingface_hub import snapshot_download
from FlagEmbedding.abc.evaluation.data_loader import AbsDataLoader

logger = logging.getLogger(__name__)


class MSMARCOADataLoader(AbsDataLoader):
    def __init__(
        self,
        dataset_dir: str,
        cache_dir: str = None,
        token: str = None,
        query_dir: str,
        rels_dir: str,
        text_type: str = 'passage',
        split: str = 'dev',
        **kwargs
    ):
        self.dataset_dir = dataset_dir
        self.query_dir = query_dir
        self.rels_dir = rels_dir
        self.cache_dir = cache_dir
        self.token = token
        
        if text_type == 'passage':
            if not os.path.exists(self.dataset_dir):
                try:
                    logger.warning(f"Trying to download dataset from huggingface hub: {dataset_dir}")
                    dataset_dir = snapshot_download(
                        repo_id='Tevatron/msmarco-passage-corpus',
                        cache_dir=os.getenv('HF_HUB_CACHE', cache_dir),
                        token=os.getenv('HF_TOKEN', token),
                        **kwargs
                    )
                    self.dataset_dir = dataset_dir
                except Exception as e:
                    logger.error(f"Error downloading dataset: {e}")

                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        


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
