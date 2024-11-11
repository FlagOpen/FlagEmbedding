import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

from .utils.normalize_text import normalize_text

logger = logging.getLogger(__name__)


class MKQAEvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for MKQA.
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return ['en', 'ar', 'fi', 'ja', 'ko', 'ru', 'es', 'sv', 'he', 'th', 'da', 'de', 'fr', 'it', 'nl', 'pl', 'pt', 'hu', 'vi', 'ms', 'km', 'no', 'tr', 'zh_cn', 'zh_hk', 'zh_tw']

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        return ["test"]

    def load_corpus(self, dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        """Load the corpus.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to None.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of corpus.
        """
        if self.dataset_dir is not None:
            # same corpus for all languages
            save_dir = self.dataset_dir
            return self._load_local_corpus(save_dir, dataset_name=dataset_name)
        else:
            return self._load_remote_corpus(dataset_name=dataset_name)

    def _load_local_qrels(self, save_dir: str, dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Try to load qrels from local datasets.

        Args:
            save_dir (str): Directory that save the data files.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.

        Raises:
            ValueError: No local qrels found, will try to download from remote.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrels.
        """
        checked_split = self.check_splits(split)
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
                qrels[qid] = data['answers']

            return datasets.DatasetDict(qrels)

    def _load_remote_corpus(
        self,
        dataset_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """
        Refer to: https://arxiv.org/pdf/2402.03216. We use the corpus from the BeIR dataset.
        """
        corpus = datasets.load_dataset(
            "BeIR/nq", "corpus",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            download_mode=self.hf_download_mode
        )["corpus"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    docid, title, text = str(data["_id"]), normalize_text(data["title"]).lower(), normalize_text(data["text"]).lower()
                    _data = {
                        "id": docid,
                        "title": title,
                        "text": text
                    }
                    corpus_dict[docid] = {
                        "title": title,
                        "text": text
                    }
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} corpus saved to {save_path}")
        else:
            corpus_dict = {}
            for data in tqdm(corpus, desc="Loading corpus"):
                docid, title, text = str(data["_id"]), normalize_text(data["title"]), normalize_text(data["text"])
                corpus_dict[docid] = {
                    "title": title,
                    "text": text
                }
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load remote qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        endpoint = f"{os.getenv('HF_ENDPOINT', 'https://huggingface.co')}/datasets/Shitao/bge-m3-data"
        queries_download_url = f"{endpoint}/resolve/main/MKQA_test-data.zip"

        qrels_save_dir = self._download_zip_file(queries_download_url, self.cache_dir)
        qrels_save_path = os.path.join(qrels_save_dir, f"{dataset_name}.jsonl")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = {}
            with open(save_path, "w", encoding="utf-8") as f1:
                with open(qrels_save_path, "r", encoding="utf-8") as f2:
                    for line in tqdm(f2.readlines(), desc="Loading and Saving qrels"):
                        data = json.loads(line)
                        qid, answers = str(data["id"]), data["answers"]
                        _data = {
                            "qid": qid,
                            "answers": answers
                        }
                        if qid not in qrels_dict:
                            qrels_dict[qid] = {}
                        qrels_dict[qid] = answers
                        f1.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} qrels saved to {save_path}")
        else:
            qrels_dict = {}
            with open(qrels_save_path, "r", encoding="utf-8") as f:
                for line in tqdm(f.readlines(), desc="Loading qrels"):
                    data = json.loads(line)
                    qid, answers = str(data["id"]), data["answers"]
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid] = answers
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        endpoint = f"{os.getenv('HF_ENDPOINT', 'https://huggingface.co')}/datasets/Shitao/bge-m3-data"
        queries_download_url = f"{endpoint}/resolve/main/MKQA_test-data.zip"

        queries_save_dir = self._download_zip_file(queries_download_url, self.cache_dir)
        queries_save_path = os.path.join(queries_save_dir, f"{dataset_name}.jsonl")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f1:
                with open(queries_save_path, "r", encoding="utf-8") as f2:
                    for line in tqdm(f2.readlines(), desc="Loading and Saving queries"):
                        data = json.loads(line)
                        qid, query = str(data["id"]), data["question"]
                        _data = {
                            "id": qid,
                            "text": query
                        }
                        queries_dict[qid] = query
                        f1.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} queries saved to {save_path}")
        else:
            queries_dict = {}
            with open(queries_save_path, "r", encoding="utf-8") as f:
                for line in tqdm(f.readlines(), desc="Loading queries"):
                    data = json.loads(line)
                    qid, query = str(data["id"]), data["question"]
                    queries_dict[qid] = query
        return datasets.DatasetDict(queries_dict)
