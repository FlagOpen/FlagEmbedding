import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class MIRACLEvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for MIRACL.
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return ["ar", "bn", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "zh", "de", "yo"]

    def available_splits(self, dataset_name: str) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        if dataset_name in ["de", "yo"]:
            return ["dev"]
        else:
            return ["train", "dev"]

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
            "miracl/miracl-corpus", dataset_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            download_mode=self.hf_download_mode
        )["train"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    docid, title, text = str(data["docid"]), data["title"], data["text"]
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
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {str(data["docid"]): {"title": data["title"], "text": data["text"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = 'dev',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'dev'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        endpoint = f"{os.getenv('HF_ENDPOINT', 'https://huggingface.co')}/datasets/miracl/miracl"
        qrels_download_url = f"{endpoint}/resolve/main/miracl-v1.0-{dataset_name}/qrels/qrels.miracl-v1.0-{dataset_name}-{split}.tsv"

        qrels_save_path = self._download_file(qrels_download_url, self.cache_dir)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = {}
            with open(save_path, "w", encoding="utf-8") as f1:
                with open(qrels_save_path, "r", encoding="utf-8") as f2:
                    for line in tqdm(f2.readlines(), desc="Loading and Saving qrels"):
                        qid, _, docid, rel = line.strip().split("\t")
                        qid, docid, rel = str(qid), str(docid), int(rel)
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": rel
                        }
                        if qid not in qrels_dict:
                            qrels_dict[qid] = {}
                        qrels_dict[qid][docid] = rel
                        f1.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} qrels saved to {save_path}")
        else:
            qrels_dict = {}
            with open(qrels_save_path, "r", encoding="utf-8") as f:
                for line in tqdm(f.readlines(), desc="Loading qrels"):
                    qid, _, docid, rel = line.strip().split("\t")
                    qid, docid, rel = str(qid), str(docid), int(rel)
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid][docid] = rel
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = 'dev',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'dev'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        endpoint = f"{os.getenv('HF_ENDPOINT', 'https://huggingface.co')}/datasets/miracl/miracl"
        queries_download_url = f"{endpoint}/resolve/main/miracl-v1.0-{dataset_name}/topics/topics.miracl-v1.0-{dataset_name}-{split}.tsv"

        queries_save_path = self._download_file(queries_download_url, self.cache_dir)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f1:
                with open(queries_save_path, "r", encoding="utf-8") as f2:
                    for line in tqdm(f2.readlines(), desc="Loading and Saving queries"):
                        qid, query = line.strip().split("\t")
                        qid = str(qid)
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
                    qid, query = line.strip().split("\t")
                    qid = str(qid)
                    queries_dict[qid] = query
        return datasets.DatasetDict(queries_dict)
