import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class BrightShortEvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for Bright(short).
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return [
            # StackExchange
            "biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living",
            # Coding
            "leetcode", "pony",
            # Theorem-based
            "aops", "theoremqa_questions", "theoremqa_theorems"
        ]

    def available_splits(self, dataset_name: str) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ]

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
            "xlangai/bright", "documents",
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    docid, text = str(data["id"]), data["content"]
                    _data = {
                        "id": docid,
                        "text": text
                    }
                    corpus_dict[docid] = {"text": text}
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {str(data["id"]): {"text": data["content"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = 'examples',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'examples'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        examples = datasets.load_dataset(
            "xlangai/bright", split,
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = defaultdict(dict)
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(examples, desc="Loading and Saving qrels"):

                    # NOTE: we modify the qid here to distinguish the queries from different splits
                    qid = f'{split}-{data["id"]}'

                    for docid in data["gold_ids"]:
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": 1
                        }
                        qrels_dict[qid][docid] = 1
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")

                    # NOTE: we record the excluded_ids in qrels with relevance 0 to remove corresponding documents from raw search results. Refer to `searcher.py` for details.
                    for ex_docid in list(set(data["excluded_ids"])):
                        if ex_docid == "N/A":
                            continue
                        assert ex_docid not in qrels_dict[qid], f"{ex_docid} in {qid}"
                        _data = {
                            "qid": qid,
                            "docid": ex_docid,
                            "relevance": 0
                        }
                        qrels_dict[qid][ex_docid] = 0
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
        else:
            qrels_dict = defaultdict(dict)
            for data in tqdm(examples, desc="Loading qrels"):

                # NOTE: we modify the qid here to distinguish the queries from different splits
                qid = f'{split}-{data["id"]}'

                for docid in data["gold_ids"]:
                    qrels_dict[qid][docid] = 1

                # NOTE: we record the excluded_ids in qrels with relevance 0 to remove corresponding documents from raw search results. Refer to `searcher.py` for details.
                for ex_docid in data["excluded_ids"]:
                    if ex_docid == "N/A":
                        continue
                    assert ex_docid not in qrels_dict[qid], f"{ex_docid} in {qid}"
                    _data = {
                        "qid": qid,
                        "docid": ex_docid,
                        "relevance": 0
                    }
                    qrels_dict[qid][ex_docid] = 0
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = 'examples',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'examples'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        examples = datasets.load_dataset(
            "xlangai/bright", split,
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(examples, desc="Loading and Saving queries"):

                    # NOTE: we modify the qid here to distinguish the queries from different splits
                    qid, query = f'{split}-{data["id"]}', data["query"]

                    _data = {
                        "id": qid,
                        "text": query
                    }
                    queries_dict[qid] = query
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
        else:
            # NOTE: we modify the qid here to distinguish the queries from different splits
            queries_dict = {f'{split}-{data["id"]}': data["query"] for data in tqdm(examples, desc="Loading queries")}
        return datasets.DatasetDict(queries_dict)


class BrightLongEvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for Bright(long).
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return [
            # StackExchange
            "biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living",
            # Coding
            "pony",
        ]

    def available_splits(self, dataset_name: str) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ]

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
            "xlangai/bright", "long_documents",
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    docid, text = str(data["id"]), data["content"]
                    _data = {
                        "id": docid,
                        "text": text
                    }
                    corpus_dict[docid] = {"text": text}
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {str(data["id"]): {"text": data["content"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: str,
        split: str = 'examples',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'examples'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        examples = datasets.load_dataset(
            "xlangai/bright", split,
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = defaultdict(dict)
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(examples, desc="Loading and Saving qrels"):

                    # NOTE: we modify the qid here to distinguish the queries from different splits
                    qid = f'{split}-{data["id"]}'

                    for docid in data["gold_ids_long"]:
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": 1
                        }
                        qrels_dict[qid][docid] = 1
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")

                    # NOTE: we record the excluded_ids in qrels with relevance 0 to remove corresponding documents from raw search results. Refer to `searcher.py` for details.
                    for ex_docid in list(set(data["excluded_ids"])):
                        if ex_docid == "N/A":
                            continue
                        assert ex_docid not in qrels_dict[qid], f"{ex_docid} in {qid}"
                        _data = {
                            "qid": qid,
                            "docid": ex_docid,
                            "relevance": 0
                        }
                        qrels_dict[qid][ex_docid] = 0
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
        else:
            qrels_dict = defaultdict(dict)
            for data in tqdm(examples, desc="Loading qrels"):

                # NOTE: we modify the qid here to distinguish the queries from different splits
                qid = f'{split}-{data["id"]}'

                for docid in data["gold_ids_long"]:
                    qrels_dict[qid][docid] = 1

                # NOTE: we record the excluded_ids in qrels with relevance 0 to remove corresponding documents from raw search results. Refer to `searcher.py` for details.
                for ex_docid in data["excluded_ids"]:
                    if ex_docid == "N/A":
                        continue
                    assert ex_docid not in qrels_dict[qid], f"{ex_docid} in {qid}"
                    _data = {
                        "qid": qid,
                        "docid": ex_docid,
                        "relevance": 0
                    }
                    qrels_dict[qid][ex_docid] = 0
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: str,
        split: str = 'examples',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'examples'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        examples = datasets.load_dataset(
            "xlangai/bright", split,
            cache_dir=self.cache_dir,
            download_mode=self.hf_download_mode
        )[dataset_name]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(examples, desc="Loading and Saving queries"):

                    # NOTE: we modify the qid here to distinguish the queries from different splits
                    qid, query = f'{split}-{data["id"]}', data["query"]

                    _data = {
                        "id": qid,
                        "text": query
                    }
                    queries_dict[qid] = query
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
        else:
            # NOTE: we modify the qid here to distinguish the queries from different splits
            queries_dict = {f'{split}-{data["id"]}': data["query"] for data in tqdm(examples, desc="Loading queries")}
        return datasets.DatasetDict(queries_dict)
