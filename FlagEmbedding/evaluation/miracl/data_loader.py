import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class MIRACLEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return ["ar", "bn", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "zh", "de", "yo"]

    def available_splits(self, dataset_name: str = None) -> List[str]:
        if dataset_name in ["de", "yo"]:
            return ["dev"]
        else:
            return ["train", "dev"]

    def _load_remote_corpus(
        self,
        dataset_name: str,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        corpus = datasets.load_dataset(
            "miracl/miracl-corpus", dataset_name,
            split="train",
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    _data = {
                        "id": data["docid"],
                        "title": data["title"],
                        "text": data["text"]
                    }
                    corpus_dict[data["docid"]] = {
                        "title": data["title"],
                        "text": data["text"]
                    }
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            corpus_dict = {data["docid"]: {"title": data["title"], "text": data["text"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _download_file(self, download_url: str, save_path: str):
        cmd = f"wget -O {save_path} {download_url}"
        os.system(cmd)

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Failed to download file from {download_url} to {save_path}")
        else:
            logger.info(f"Downloaded file from {download_url} to {save_path}")

    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'dev',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        qrels_download_url = f"https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{dataset_name}/qrels/qrels.miracl-v1.0-{dataset_name}-{split}.tsv"
        qrels_save_path = os.path.join(self.cache_dir, f"qrels.miracl-v1.0-{dataset_name}-{split}.tsv")

        self._download_file(qrels_download_url, qrels_save_path)

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
                    qid, docid, rel = int(qid), int(docid), int(rel)
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid][docid] = rel
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        queries_download_url = f"https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-{dataset_name}/topics/topics.miracl-v1.0-{dataset_name}-{split}.tsv"
        queries_save_path = os.path.join(self.cache_dir, f"topics.miracl-v1.0-{dataset_name}-{split}.tsv")

        self._download_file(queries_download_url, queries_save_path)

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
