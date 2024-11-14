import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class MSMARCOEvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for MSMARCO.
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return ["passage", "document"]

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (Optional[str], optional): Dataset name. Defaults to ``None``.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        return ["dev", "dl19", "dl20"]

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
        if dataset_name == 'passage':
            corpus = datasets.load_dataset(
                'Tevatron/msmarco-passage-corpus', 
                'default', 
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                download_mode=self.hf_download_mode
            )['train']
        else:
            corpus = datasets.load_dataset(
                'irds/msmarco-document', 
                'docs', 
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                download_mode=self.hf_download_mode
            )

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "corpus.jsonl")
            corpus_dict = {}
            with open(save_path, "w", encoding="utf-8") as f:
                for data in tqdm(corpus, desc="Loading and Saving corpus"):
                    if dataset_name == 'passage':
                        _data = {
                            "id": data["docid"],
                            "title": data["title"],
                            "text": data["text"]
                        }
                        corpus_dict[data["docid"]] = {
                            "title": data["title"],
                            "text": data["text"]
                        }
                    else:
                        _data = {
                            "id": data["doc_id"],
                            "title": data["title"],
                            "text": data["body"]
                        }
                        corpus_dict[data["doc_id"]] = {
                            "title": data["title"],
                            "text": data["body"]
                        }
                    f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
        else:
            if dataset_name == 'passage':
                corpus_dict = {data["docid"]: {"title": data["title"], "text": data["text"]} for data in tqdm(corpus, desc="Loading corpus")}
            else:
                corpus_dict = {data["doc_id"]: {"title": data["title"], "text": data["body"]} for data in tqdm(corpus, desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
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
        if dataset_name == 'passage':
            if split == 'dev':
                qrels = datasets.load_dataset(
                    'BeIR/msmarco-qrels', 
                    split='validation',
                    trust_remote_code=True,
                    cache_dir=self.cache_dir,
                    download_mode=self.hf_download_mode
                )
                qrels_download_url = None
            elif split == 'dl19':
                qrels_download_url = "https://trec.nist.gov/data/deep/2019qrels-pass.txt"
            else:
                qrels_download_url = "https://trec.nist.gov/data/deep/2020qrels-pass.txt"
        else:
            if split == 'dev':
                qrels_download_url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"
            elif split == 'dl19':
                qrels_download_url = "https://trec.nist.gov/data/deep/2019qrels-docs.txt"
            else:
                qrels_download_url = "https://trec.nist.gov/data/deep/2020qrels-docs.txt"

        if qrels_download_url is not None:
            qrels_save_path = self._download_file(qrels_download_url, self.cache_dir)
        else:
            qrels_save_path = None
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
            qrels_dict = {}
            if qrels_save_path is not None:
                with open(save_path, "w", encoding="utf-8") as f1:
                    with open(qrels_save_path, "r", encoding="utf-8") as f2:
                        for line in tqdm(f2.readlines(), desc="Loading and Saving qrels"):
                            qid, _, docid, rel = line.strip().split()
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
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    for data in tqdm(qrels, desc="Loading and Saving qrels"):
                        qid, docid, rel = str(data['query-id']), str(data['corpus-id']), int(data['score'])
                        _data = {
                            "qid": qid,
                            "docid": docid,
                            "relevance": rel
                        }
                        if qid not in qrels_dict:
                            qrels_dict[qid] = {}
                        qrels_dict[qid][docid] = rel
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} qrels saved to {save_path}")
        else:
            qrels_dict = {}
            if qrels_save_path is None:
                with open(qrels_save_path, "r", encoding="utf-8") as f:
                    for line in tqdm(f.readlines(), desc="Loading qrels"):
                        qid, _, docid, rel = line.strip().split()
                        qid, docid, rel = str(qid), str(docid), int(rel)
                        if qid not in qrels_dict:
                            qrels_dict[qid] = {}
                        qrels_dict[qid][docid] = rel
            else:
                for data in tqdm(qrels, desc="Loading queries"):
                    qid, docid, rel = str(data['query-id']), str(data['corpus-id']), int(data['score'])
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
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            split (str, optional): Split of the dataset. Defaults to ``'test'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        if split == 'dev':
            if dataset_name == 'passage':
                queries = datasets.load_dataset(
                    'BeIR/msmarco', 
                    'queries',
                    trust_remote_code=True,
                    cache_dir=self.cache_dir,
                    download_mode=self.hf_download_mode
                )['queries']
                queries_save_path = None
            else:
                queries_download_url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"
                queries_save_path = self._download_gz_file(queries_download_url, self.cache_dir)
        else:
            year = split.replace("dl", "")
            queries_download_url = f"https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test20{year}-queries.tsv.gz"
            queries_save_path = self._download_gz_file(queries_download_url, self.cache_dir)

        qrels = self.load_qrels(dataset_name=dataset_name, split=split)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
            queries_dict = {}
            if queries_save_path is not None:
                with open(save_path, "w", encoding="utf-8") as f1:
                    with open(queries_save_path, "r", encoding="utf-8") as f2:
                        for line in tqdm(f2.readlines(), desc="Loading and Saving queries"):
                            qid, query = line.strip().split("\t")
                            if qid not in qrels.keys(): continue
                            qid = str(qid)
                            _data = {
                                "id": qid,
                                "text": query
                            }
                            queries_dict[qid] = query
                            f1.write(json.dumps(_data, ensure_ascii=False) + "\n")
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    for data in tqdm(queries, desc="Loading and Saving queries"):
                        qid, query = data['_id'], data['text']
                        if qid not in qrels.keys(): continue
                        _data = {
                            "id": qid,
                            "text": query
                        }
                        queries_dict[qid] = query
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
            logging.info(f"{self.eval_name} {dataset_name} queries saved to {save_path}")
        else:
            queries_dict = {}
            if queries_save_path is not None:
                with open(queries_save_path, "r", encoding="utf-8") as f:
                    for line in tqdm(f.readlines(), desc="Loading queries"):
                        qid, query = line.strip().split("\t")
                        qid = str(qid)
                        if qid not in qrels.keys(): continue
                        queries_dict[qid] = query
            else:
                for data in tqdm(queries, desc="Loading queries"):
                    qid, query = data['_id'], data['text']
                    if qid not in qrels.keys(): continue
                    queries_dict[qid] = query
        return datasets.DatasetDict(queries_dict)
