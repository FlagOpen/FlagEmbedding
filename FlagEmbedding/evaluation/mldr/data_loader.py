import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class MLDREvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for MLDR.
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "th", "zh"]

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (Optional[str], optional): Dataset name. Defaults to ``None``.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        return ["train", "dev", "test"]

    def _load_remote_corpus(
        self,
        dataset_name: str,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the corpus dataset from HF.

        Args:
            dataset_name (str): Name of the dataset.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of corpus.
        """
        corpus = datasets.load_dataset(
            "Shitao/MLDR", f"corpus-{dataset_name}",
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
                    docid, text = str(data["docid"]), data["text"]
                    _data = {
                        "id": docid,
                        "text": text
                    }
                    corpus_dict[docid] = {"text": text}
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {str(data["docid"]): {"text": data["text"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = "test",
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        qrels_data = datasets.load_dataset(
            "Shitao/MLDR", dataset_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            download_mode=self.hf_download_mode
        )[split]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(qrels_data, desc="Loading and Saving qrels"):
                    qid = str(data["query_id"])
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    for doc in data["positive_passages"]:
                        docid = str(doc["docid"])
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": 1
                        }
                        qrels_dict[qid][docid] = 1
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
                    for doc in data["negative_passages"]:
                        docid = str(doc["docid"])
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": 0
                        }
                        qrels_dict[qid][docid] = 0
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} qrels saved to {save_path}")
        else:
            qrels_dict = {}
            for data in tqdm(qrels_data, desc="Loading qrels"):
                qid = str(data["query_id"])
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                for doc in data["positive_passages"]:
                    docid = str(doc["docid"])
                    qrels_dict[qid][docid] = 1
                for doc in data["negative_passages"]:
                    docid = str(doc["docid"])
                    qrels_dict[qid][docid] = 0
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = "test",
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
        queries_data = datasets.load_dataset(
            "Shitao/MLDR", dataset_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            download_mode=self.hf_download_mode
        )[split]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(queries_data, desc="Loading and Saving queries"):
                    qid, query = str(data["query_id"]), data["query"]
                    _data = {
                        "id": qid,
                        "text": query
                    }
                    queries_dict[qid] = query
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} queries saved to {save_path}")
        else:
            queries_dict = {}
            for data in tqdm(queries_data, desc="Loading queries"):
                qid, query = str(data["query_id"]), data["query"]
                queries_dict[qid] = query
        return datasets.DatasetDict(queries_dict)
