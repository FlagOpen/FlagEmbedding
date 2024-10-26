"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/data_loader.py
"""
import os
import logging
import datasets
from abc import ABC, abstractmethod
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class AbsEvalDataLoader(ABC):
    def __init__(
        self,
        eval_name: str,
        dataset_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.eval_name = eval_name
        self.dataset_dir = dataset_dir
        if cache_dir is None:
            cache_dir = os.getenv('HF_HUB_CACHE', '~/.cache/huggingface/hub')
        self.cache_dir = os.path.join(cache_dir, eval_name)
        self.token = token

    def available_dataset_names(self) -> List[str]:
        """
        Returns: List[str]: Available dataset names.
        """
        return []

    @abstractmethod
    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Returns: List[str]: Available splits in the dataset.
        """
        pass

    def check_dataset_names(self, dataset_names: Union[str, List[str]]) -> List[str]:
        available_dataset_names = self.available_dataset_names()
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        for dataset_name in dataset_names:
            if dataset_name not in available_dataset_names:
                raise ValueError(f"Dataset name '{dataset_name}' not found in the dataset. Available dataset names: {available_dataset_names}")
        return dataset_names

    def check_splits(self, splits: Union[str, List[str]], dataset_name: Optional[str] = None) -> List[str]:
        available_splits = self.available_splits(dataset_name=dataset_name)
        if isinstance(splits, str):
            splits = [splits]
        checked_splits = []
        for split in splits:
            if split not in available_splits:
                logger.warning(f"Split '{split}' not found in the dataset. Removing it from the list.")
            else:
                checked_splits.append(split)
        return checked_splits

    def load_corpus(self, dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        if self.dataset_dir is not None:
            if dataset_name is None:
                save_dir = self.dataset_dir
            else:
                save_dir = os.path.join(self.dataset_dir, dataset_name)
            return self._load_local_corpus(save_dir, dataset_name=dataset_name)
        else:
            return self._load_remote_corpus(dataset_name=dataset_name)

    def load_qrels(self, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        if self.dataset_dir is not None:
            if dataset_name is None:
                save_dir = self.dataset_dir
            else:
                checked_dataset_names = self.check_dataset_names(dataset_name)
                if len(checked_dataset_names) == 0:
                    raise ValueError(f"Dataset name {dataset_name} not found in the dataset.")
                dataset_name = checked_dataset_names[0]

                save_dir = os.path.join(self.dataset_dir, dataset_name)

            return self._load_local_qrels(save_dir, dataset_name=dataset_name, split=split)
        else:
            return self._load_remote_qrels(dataset_name=dataset_name, split=split)

    def load_queries(self, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        if self.dataset_dir is not None:
            if dataset_name is None:
                save_dir = self.dataset_dir
            else:
                checked_dataset_names = self.check_dataset_names(dataset_name)
                if len(checked_dataset_names) == 0:
                    raise ValueError(f"Dataset name {dataset_name} not found in the dataset.")
                dataset_name = checked_dataset_names[0]

                save_dir = os.path.join(self.dataset_dir, dataset_name)

            return self._load_local_queries(save_dir, dataset_name=dataset_name, split=split)
        else:
            return self._load_remote_queries(dataset_name=dataset_name, split=split)

    def _load_remote_corpus(
        self,
        dataset_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        raise NotImplementedError("Loading remote corpus is not implemented.")

    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        raise NotImplementedError("Loading remote qrels is not implemented.")

    def _load_remote_queries(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        raise NotImplementedError("Loading remote queries is not implemented.")

    def _load_local_corpus(self, save_dir: str, dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        corpus_path = os.path.join(save_dir, 'corpus.jsonl')
        if not os.path.exists(corpus_path):
            logger.warning(f"Corpus not found in {corpus_path}. Trying to download the corpus from the remote and save it to {save_dir}.")
            return self._load_remote_corpus(dataset_name=dataset_name, save_dir=save_dir)
        else:
            corpus_data = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']

            corpus = {e['id']: {'text': e['text']} for e in corpus_data}
            return datasets.DatasetDict(corpus)

    def _load_local_qrels(self, save_dir: str, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        checked_split = self.check_splits(split)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]

        qrels_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
        if not os.path.exists(qrels_path):
            logger.warning(f"Qrels not found in {qrels_path}. Trying to download the qrels from the remote and save it to {save_dir}.")
            return self._load_remote_qrels(dataset_name=dataset_name, split=split, save_dir=save_dir)
        else:
            qrels_data = datasets.load_dataset('json', data_files=qrels_path, cache_dir=self.cache_dir)['train']

            qrels = {}
            for data in qrels_data:
                qid = data['qid']
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][data['docid']] = data['relevance']

            return datasets.DatasetDict(qrels)

    def _load_local_queries(self, save_dir: str, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        checked_split = self.check_splits(split)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]

        queries_path = os.path.join(save_dir, f"{split}_queries.jsonl")
        if not os.path.exists(queries_path):
            logger.warning(f"Queries not found in {queries_path}. Trying to download the queries from the remote and save it to {save_dir}.")
            return self._load_remote_queries(dataset_name=dataset_name, split=split, save_dir=save_dir)
        else:
            queries_data = datasets.load_dataset('json', data_files=queries_path, cache_dir=self.cache_dir)['train']

            queries = {e['id']: e['text'] for e in queries_data}
            return datasets.DatasetDict(queries)
