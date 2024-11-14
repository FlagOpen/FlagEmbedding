import os
import json
import logging
import datasets
from tqdm import tqdm
from typing import List, Optional
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class BEIREvalDataLoader(AbsEvalDataLoader):
    """
    Data loader class for BEIR.
    """
    def available_dataset_names(self) -> List[str]:
        """
        Get the available dataset names.

        Returns:
            List[str]: All the available dataset names.
        """
        return ['arguana', 'climate-fever', 'cqadupstack', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact', 'trec-covid', 'webis-touche2020']

    def available_sub_dataset_names(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get the available sub-dataset names.

        Args:
            dataset_name (Optional[str], optional): All the available sub-dataset names. Defaults to ``None``.

        Returns:
            List[str]: All the available sub-dataset names.
        """
        if dataset_name == 'cqadupstack':
            return ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
        return None

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get the avaialble splits.

        Args:
            dataset_name (str): Dataset name.

        Returns:
            List[str]: All the available splits for the dataset.
        """
        if dataset_name == 'msmarco':
            return ['dev']
        return ['test']

    def _load_remote_corpus(
        self,
        dataset_name: str,
        sub_dataset_name: Optional[str] = None,
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the corpus dataset from HF.

        Args:
            dataset_name (str): Name of the dataset.
            sub_dataset_name (Optional[str]): Name of the sub-dataset. Defaults to ``None``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of corpus.
        """
        if dataset_name != 'cqadupstack':
            corpus = datasets.load_dataset(
                'BeIR/{d}'.format(d=dataset_name),
                'corpus',
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                download_mode=self.hf_download_mode
            )['corpus']

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "corpus.jsonl")
                corpus_dict = {}
                with open(save_path, "w", encoding="utf-8") as f:
                    for data in tqdm(corpus, desc="Loading and Saving corpus"):
                        _data = {
                            "id": data["_id"],
                            "title": data["title"],
                            "text": data["text"]
                        }
                        corpus_dict[data["_id"]] = {
                            "title": data["title"],
                            "text": data["text"]
                        }
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
                logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
            else:
                corpus_dict = {data["docid"]: {"title": data["title"], "text": data["text"]} for data in tqdm(corpus, desc="Loading corpus")}
        else:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
            data_path = util.download_and_unzip(url, self.cache_dir)
            full_path = os.path.join(data_path, sub_dataset_name)
            corpus, _, _ = GenericDataLoader(data_folder=full_path).load(split="test")
            if save_dir is not None:
                new_save_dir = os.path.join(save_dir, sub_dataset_name)
                os.makedirs(new_save_dir, exist_ok=True)
                save_path = os.path.join(new_save_dir, "corpus.jsonl")
                corpus_dict = {}
                with open(save_path, "w", encoding="utf-8") as f:
                    for _id in tqdm(corpus.keys(), desc="Loading corpus"):
                        _data = {
                            "id": _id,
                            "title": corpus[_id]["title"],
                            "text": corpus[_id]["text"]
                        }
                        corpus_dict[_id] = {
                            "title": corpus[_id]["title"],
                            "text": corpus[_id]["text"]
                        }
                        f.write(json.dumps(_data, ensure_ascii=False) + "\n")
                logging.info(f"{self.eval_name} {dataset_name} corpus saved to {save_path}")
            else:
                corpus_dict = {_id: {"title": corpus[_id]["title"], "text": corpus[_id]["text"]} for _id in tqdm(corpus.keys(), desc="Loading corpus")}
        return datasets.DatasetDict(corpus_dict)

    def _load_remote_qrels(
        self,
        dataset_name: Optional[str] = None,
        sub_dataset_name: Optional[str] = None,
        split: str = 'dev',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the qrels from HF.

        Args:
            dataset_name (str): Name of the dataset.
            sub_dataset_name (Optional[str]): Name of the sub-dataset. Defaults to ``None``.
            split (str, optional): Split of the dataset. Defaults to ``'dev'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of qrel.
        """
        if dataset_name != 'cqadupstack':
            qrels = datasets.load_dataset(
                'BeIR/{d}-qrels'.format(d=dataset_name),
                split=split, 
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                download_mode=self.hf_download_mode
            )

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
                qrels_dict = {}
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
                for data in tqdm(qrels, desc="Loading queries"):
                    qid, docid, rel = str(data['query-id']), str(data['corpus-id']), int(data['score'])
                    if qid not in qrels_dict:
                        qrels_dict[qid] = {}
                    qrels_dict[qid][docid] = rel
        else:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
            data_path = util.download_and_unzip(url, self.cache_dir)
            full_path = os.path.join(data_path, sub_dataset_name)
            _, _, qrels = GenericDataLoader(data_folder=full_path).load(split="test")
            if save_dir is not None:
                new_save_dir = os.path.join(save_dir, sub_dataset_name)
                os.makedirs(new_save_dir, exist_ok=True)
                save_path = os.path.join(new_save_dir, f"{split}_qrels.jsonl")
                qrels_dict = {}
                with open(save_path, "w", encoding="utf-8") as f:
                    for qid in tqdm(qrels.keys(), desc="Loading and Saving qrels"):
                        for docid in tqdm(qrels[qid].keys()):
                            rel = int(qrels[qid][docid])
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
                for qid in tqdm(qrels.keys(), desc="Loading qrels"):
                    for docid in tqdm(qrels[qid].keys()):
                        rel = int(qrels[qid][docid])
                        if qid not in qrels_dict:
                            qrels_dict[qid] = {}
                        qrels_dict[qid][docid] = rel
        return datasets.DatasetDict(qrels_dict)

    def _load_remote_queries(
        self,
        dataset_name: Optional[str] = None,
        sub_dataset_name: Optional[str] = None,
        split: str = 'test',
        save_dir: Optional[str] = None
    ) -> datasets.DatasetDict:
        """Load the queries from HF.

        Args:
            dataset_name (str): Name of the dataset.
            sub_dataset_name (Optional[str]): Name of the sub-dataset. Defaults to ``None``.
            split (str, optional): Split of the dataset. Defaults to ``'dev'``.
            save_dir (Optional[str], optional): Directory to save the dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: Loaded datasets instance of queries.
        """
        qrels = self.load_qrels(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)

        if dataset_name != 'cqadupstack':
            queries = datasets.load_dataset(
                'BeIR/{d}'.format(d=dataset_name), 
                'queries', 
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                download_mode=self.hf_download_mode
            )['queries']

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{split}_queries.jsonl")
                queries_dict = {}
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
                for data in tqdm(queries, desc="Loading queries"):
                    qid, query = data['_id'], data['text']
                    if qid not in qrels.keys(): continue
                    queries_dict[qid] = query
        else:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
            data_path = util.download_and_unzip(url, self.cache_dir)
            full_path = os.path.join(data_path, sub_dataset_name)
            _, queries, _ = GenericDataLoader(data_folder=full_path).load(split="test")
            if save_dir is not None:
                new_save_dir = os.path.join(save_dir, sub_dataset_name)
                os.makedirs(new_save_dir, exist_ok=True)
                save_path = os.path.join(new_save_dir, f"{split}_queries.jsonl")
                queries_dict = {}
                with open(save_path, "w", encoding="utf-8") as f:
                    for qid in tqdm(queries.keys(), desc="Loading and Saving queries"):
                        query = queries[qid]
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
                for qid in tqdm(queries.keys(), desc="Loading queries"):
                    query = queries[qid]
                    if qid not in qrels.keys(): continue
                    queries_dict[qid] = query
        return datasets.DatasetDict(queries_dict)

    def load_corpus(self, dataset_name: Optional[str] = None, sub_dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        """Load the corpus from the dataset.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: A dict of corpus with id as key, title and text as value.
        """
        if self.dataset_dir is not None:
            if dataset_name is None:
                save_dir = self.dataset_dir
            else:
                save_dir = os.path.join(self.dataset_dir, dataset_name)
            return self._load_local_corpus(save_dir, dataset_name=dataset_name, sub_dataset_name=sub_dataset_name)
        else:
            return self._load_remote_corpus(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name)

    def load_qrels(self, dataset_name: Optional[str] = None, sub_dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load the qrels from the dataset.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.
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

            return self._load_local_qrels(save_dir, dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)
        else:
            return self._load_remote_qrels(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)

    def load_queries(self, dataset_name: Optional[str] = None, sub_dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load the queries from the dataset.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.
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

            return self._load_local_queries(save_dir, dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)
        else:
            return self._load_remote_queries(dataset_name=dataset_name, sub_dataset_name=sub_dataset_name, split=split)

    def _load_local_corpus(self, save_dir: str, dataset_name: Optional[str] = None, sub_dataset_name: Optional[str] = None) -> datasets.DatasetDict:
        """Load corpus from local dataset.

        Args:
            save_dir (str): Path to save the loaded corpus.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.

        Returns:
            datasets.DatasetDict: A dict of corpus with id as key, title and text as value.
        """
        if sub_dataset_name is None:
            corpus_path = os.path.join(save_dir, 'corpus.jsonl')
        else:
            corpus_path = os.path.join(save_dir, sub_dataset_name, 'corpus.jsonl')
        if self.force_redownload or not os.path.exists(corpus_path):
            logger.warning(f"Corpus not found in {corpus_path}. Trying to download the corpus from the remote and save it to {save_dir}.")
            return self._load_remote_corpus(dataset_name=dataset_name, save_dir=save_dir, sub_dataset_name=sub_dataset_name)
        else:
            if sub_dataset_name is not None:
                save_dir = os.path.join(save_dir, sub_dataset_name)
            corpus_data = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']

            corpus = {}
            for e in corpus_data:
                corpus[e['id']] = {'title': e.get('title', ""), 'text': e['text']}

            return datasets.DatasetDict(corpus)

    def _load_local_qrels(self, save_dir: str, dataset_name: Optional[str] = None, sub_dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load relevance from local dataset.

        Args:
            save_dir (str):  Path to save the loaded relevance.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.
            split (str, optional): Split to load from the local dataset. Defaults to ``'test'``.

        Raises:
            ValueError

        Returns:
            datasets.DatasetDict: A dict of relevance of query and document.
        """
        checked_split = self.check_splits(split)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]

        if sub_dataset_name is None:
            qrels_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
        else:
            qrels_path = os.path.join(save_dir, sub_dataset_name, f"{split}_qrels.jsonl")
        if self.force_redownload or not os.path.exists(qrels_path):
            logger.warning(f"Qrels not found in {qrels_path}. Trying to download the qrels from the remote and save it to {save_dir}.")
            return self._load_remote_qrels(dataset_name=dataset_name, split=split, sub_dataset_name=sub_dataset_name, save_dir=save_dir)
        else:
            if sub_dataset_name is not None:
                save_dir = os.path.join(save_dir, sub_dataset_name)
            qrels_data = datasets.load_dataset('json', data_files=qrels_path, cache_dir=self.cache_dir)['train']

            qrels = {}
            for data in qrels_data:
                qid = data['qid']
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][data['docid']] = data['relevance']

            return datasets.DatasetDict(qrels)

    def _load_local_queries(self, save_dir: str, dataset_name: Optional[str] = None, sub_dataset_name: Optional[str] = None, split: str = 'test') -> datasets.DatasetDict:
        """Load queries from local dataset.

        Args:
            save_dir (str):  Path to save the loaded queries.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to ``None``.
            sub_dataset_name (Optional[str], optional): Name of the sub-dataset. Defaults to ``None``.
            split (str, optional): Split to load from the local dataset. Defaults to ``'test'``.

        Raises:
            ValueError

        Returns:
            datasets.DatasetDict: A dict of queries with id as key, query text as value.
        """
        checked_split = self.check_splits(split)
        if len(checked_split) == 0:
            raise ValueError(f"Split {split} not found in the dataset.")
        split = checked_split[0]

        if sub_dataset_name is None:
            queries_path = os.path.join(save_dir, f"{split}_queries.jsonl")
        else:
            queries_path = os.path.join(save_dir, sub_dataset_name, f"{split}_queries.jsonl")
        if self.force_redownload or not os.path.exists(queries_path):
            logger.warning(f"Queries not found in {queries_path}. Trying to download the queries from the remote and save it to {save_dir}.")
            return self._load_remote_queries(dataset_name=dataset_name, split=split, sub_dataset_name=sub_dataset_name, save_dir=save_dir)
        else:
            if sub_dataset_name is not None:
                save_dir = os.path.join(save_dir, sub_dataset_name)
            queries_data = datasets.load_dataset('json', data_files=queries_path, cache_dir=self.cache_dir)['train']

            queries = {e['id']: e['text'] for e in queries_data}
            return datasets.DatasetDict(queries)
