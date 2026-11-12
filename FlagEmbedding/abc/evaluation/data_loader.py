"""
Adapted from https://github.com/AIR-Bench/AIR-Bench/blob/0.1.0/air_benchmark/evaluation_utils/data_loader.py
"""
import os
import logging
import datasets
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class AbsEvalDataLoader(ABC):
    """
    Base class of data loader for evaluation.

    Args:
        eval_name (str): The experiment name of current evaluation.
        dataset_dir (str, optional): path to the datasets. Defaults to ``None``.
        cache_dir (str, optional): Path to HuggingFace cache directory. Defaults to ``None``.
        token (str, optional): HF_TOKEN to access the private datasets/models in HF. Defaults to ``None``.
        force_redownload: If True, will force redownload the dataset to cover the local dataset. Defaults to ``False``.
    """
    def __init__(
        self,
        eval_name: str,
        dataset_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_redownload: bool = False
    ):
        self.eval_name = eval_name
        self.dataset_dir = dataset_dir
        if cache_dir is None:
            cache_dir = os.getenv('HF_HUB_CACHE', '~/.cache/huggingface/hub')
        self.cache_dir = os.path.join(cache_dir, eval_name)
        self.token = token
        self.force_redownload = force_redownload
        self.hf_download_mode = None if not force_redownload else "force_redownload"

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
        """Check the validity of dataset names

        Args:
            dataset_names (Union[str, List[str]]): a dataset name (str) or a list of dataset names (List[str])

        Raises:
            ValueError

        Returns:
            List[str]: List of valid dataset names.
        """
        available_dataset_names = self.available_dataset_names()
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        for dataset_name in dataset_names:
            if dataset_name not in available_dataset_names:
                raise ValueError(f"Dataset name '{dataset_name}' not found in the dataset. Available dataset names: {available_dataset_names}")
        return dataset_names

    def check_splits(self, splits: Union[str, List[str]], dataset_name: Optional[str] = None) -> List[str]:
        """Check whether the splits are available in the dataset.

        Args:
            splits (Union[str, List[str]]): Splits to check.
            dataset_name (Optional[str], optional): Name of dataset to check. Defaults to ``None``.

        Returns:
            List[str]: The available splits.
        """
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
        """Load the corpus from the dataset.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: A dict of corpus with id as key, title and text as value.
        """
        if self.dataset_dir is not None:
            if dataset_name is None:
                save_dir = self.dataset_dir
            else:
                save_dir = os.path.join(self.dataset_dir, dataset_name)
            return self._load_local_corpus(save_dir, dataset_name=dataset_name)
        else:
            return self._load_remote_corpus(dataset_name=dataset_name)

    def load_qrels(self, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load the qrels from the dataset.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): The split to load relevance from. Defaults to ``'test'``.

        Raises:
            ValueError

        Returns:
            datasets.DatasetDict: A dict of relevance of query and document.
        """
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
        """Load the queries from the dataset.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): The split to load queries from. Defaults to ``'test'``.

        Raises:
            ValueError

        Returns:
            datasets.DatasetDict: A dict of queries with id as key, query text as value.
        """
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
        """Abstract method to load corpus from remote dataset, to be overrode in child class.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            save_dir (Optional[str], optional): Path to save the new downloaded corpus. Defaults to ``None``.

        Raises:
            NotImplementedError: Loading remote corpus is not implemented.

        Returns:
            datasets.DatasetDict: A dict of corpus with id as key, title and text as value.
        """
        raise NotImplementedError("Loading remote corpus is not implemented.")

    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Abstract method to load relevance from remote dataset, to be overrode in child class.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): Split to load from the remote dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Path to save the new downloaded relevance. Defaults to ``None``.

        Raises:
            NotImplementedError: Loading remote qrels is not implemented.

        Returns:
            datasets.DatasetDict: A dict of relevance of query and document.
        """
        raise NotImplementedError("Loading remote qrels is not implemented.")

    def _load_remote_queries(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Abstract method to load queries from remote dataset, to be overrode in child class.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): Split to load from the remote dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Path to save the new downloaded queries. Defaults to ``None``.

        Raises:
            NotImplementedError

        Returns:
            datasets.DatasetDict: A dict of queries with id as key, query text as value.
        """
        raise NotImplementedError("Loading remote queries is not implemented.")

    def _load_local_corpus(self, save_dir: str, dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        """Load corpus from local dataset.

        Args:
            save_dir (str): Path to save the loaded corpus.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: A dict of corpus with id as key, title and text as value.
        """
        corpus_path = os.path.join(save_dir, 'corpus.jsonl')
        if self.force_redownload or not os.path.exists(corpus_path):
            logger.warning(f"Corpus not found in {corpus_path}. Trying to download the corpus from the remote and save it to {save_dir}.")
            return self._load_remote_corpus(dataset_name=dataset_name, save_dir=save_dir)
        else:
            corpus_data = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']

            corpus = {}
            for e in corpus_data:
                corpus[e['id']] = {'title': e.get('title', ""), 'text': e['text']}

            return datasets.DatasetDict(corpus)

    def _load_local_qrels(self, save_dir: str, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load relevance from local dataset.

        Args:
            save_dir (str):  Path to save the loaded relevance.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): Split to load from the local dataset. Defaults to ``'test'``.

        Raises:
            ValueError

        Returns:
            datasets.DatasetDict: A dict of relevance of query and document.
        """
        checked_split = self.check_splits(split, dataset_name=dataset_name)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]

        qrels_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
        if self.force_redownload or not os.path.exists(qrels_path):
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
        """Load queries from local dataset.

        Args:
            save_dir (str):  Path to save the loaded queries.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): Split to load from the local dataset. Defaults to ``'test'``.

        Raises:
            ValueError

        Returns:
            datasets.DatasetDict: A dict of queries with id as key, query text as value.
        """
        checked_split = self.check_splits(split, dataset_name=dataset_name)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]

        queries_path = os.path.join(save_dir, f"{split}_queries.jsonl")
        if self.force_redownload or not os.path.exists(queries_path):
            logger.warning(f"Queries not found in {queries_path}. Trying to download the queries from the remote and save it to {save_dir}.")
            return self._load_remote_queries(dataset_name=dataset_name, split=split, save_dir=save_dir)
        else:
            queries_data = datasets.load_dataset('json', data_files=queries_path, cache_dir=self.cache_dir)['train']

            queries = {e['id']: e['text'] for e in queries_data}
            return datasets.DatasetDict(queries)

    def _download_file(self, download_url: str, save_dir: str):
        """Download file from provided URL.

        Args:
            download_url (str): Source URL of the file.
            save_dir (str): Path to the directory to save the zip file.

        Raises:
            FileNotFoundError

        Returns:
            str: The path of the downloaded file.
        """
        save_path = os.path.join(save_dir, download_url.split('/')[-1])

        if self.force_redownload or (not os.path.exists(save_path) or os.path.getsize(save_path) == 0):
            cmd = ["wget", "-O", save_path, download_url]
        else:
            cmd = ["wget", "-nc", "-O", save_path, download_url]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(e.output)

        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            raise FileNotFoundError(f"Failed to download file from {download_url} to {save_path}")
        else:
            logger.info(f"Downloaded file from {download_url} to {save_path}")
            return save_path

    def _get_fpath_size(self, fpath: str) -> int:
        """Get the total size of the files in provided path.

        Args:
            fpath (str): path of files to compute the size.

        Returns:
            int: The total size in bytes.
        """
        if not os.path.isdir(fpath):
            return os.path.getsize(fpath)
        else:
            total_size = 0
            for dirpath, _, filenames in os.walk(fpath):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            return total_size

    def _download_gz_file(self, download_url: str, save_dir: str):
        """Download and unzip the gzip file from provided URL.

        Args:
            download_url (str): Source URL of the gzip file.
            save_dir (str): Path to the directory to save the gzip file.

        Raises:
            FileNotFoundError

        Returns:
            str: The path to the file after unzip.
        """
        gz_file_path = self._download_file(download_url, save_dir)
        cmd = ["gzip", "-d", gz_file_path]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(e.output)

        file_path = gz_file_path.replace(".gz", "")
        if not os.path.exists(file_path) or self._get_fpath_size(file_path) == 0:
            raise FileNotFoundError(f"Failed to unzip file {gz_file_path}")

        return file_path

    def _download_zip_file(self, download_url: str, save_dir: str):
        """Download and unzip the zip file from provided URL.

        Args:
            download_url (str): Source URL of the zip file.
            save_dir (str): Path to the directory to save the zip file.

        Raises:
            FileNotFoundError

        Returns:
            str: The path to the file after unzip.
        """
        zip_file_path = self._download_file(download_url, save_dir)
        file_path = zip_file_path.replace(".zip", "")
        if self.force_redownload or not os.path.exists(file_path):
            cmd = ["unzip", "-o", zip_file_path, "-d", file_path]
        else:
            cmd = ["unzip", "-n", zip_file_path, "-d", file_path]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(e.output)

        if not os.path.exists(file_path) or self._get_fpath_size(file_path) == 0:
            raise FileNotFoundError(f"Failed to unzip file {zip_file_path}")

        return file_path
