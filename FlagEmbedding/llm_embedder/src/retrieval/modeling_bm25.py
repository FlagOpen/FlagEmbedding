import os
import json
import subprocess
import datasets
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm
from collections import defaultdict
from src.utils.util import clear_dir, split_file_dir_name_ext


class BM25Retriever:
    def __init__(self, anserini_dir, k1=0.9, b=0.4, **kwds) -> None:
        self.anserini_dir = anserini_dir
        self.k1 = k1
        self.b = b
    
    def _prepare_collection(self, corpus:datasets.Dataset, collection_dir, max_docs_per_file=1000000):
        clear_dir(collection_dir)

        file_index = 0
        for i, doc in enumerate(tqdm(corpus, desc="Preparing Anserini Collection")):
            text = doc["content"]
            if i % max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = os.path.join(collection_dir, 'docs{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                file_index += 1
            output_dict = {'id': i, 'contents': text}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')
        output_jsonl_file.close()
    
    def _prepare_query(self, eval_data:Union[str, datasets.Dataset], query_dir:str, max_queries_per_file=10000):
        clear_dir(query_dir)

        query_ids = []
        queries = []
        if isinstance(eval_data, str):
            with open(eval_data) as f:
                for line in tqdm(f, desc="Preparing Anserini Queries"):
                    # NOTE: repr query because it may contain newline character
                    item = json.loads(line)
                    query = repr(item["query"])[1:-1]
                    # filter out empty query
                    if len(query.strip()):
                        query_ids.append(item["query_id"])
                        queries.append(query)
        elif isinstance(eval_data, datasets.Dataset):
            for item in tqdm(eval_data, desc="Preparing Anserini Queries"):
                # NOTE: repr query because it may contain newline character
                query = repr(item["query"])[1:-1]
                # filter out empty query
                if len(query.strip()):
                    query_ids.append(item["query_id"])
                    queries.append(query)
        else:
            raise ValueError(f"Expected eval_data to be instance of str or datasets.Dataset, got {type(eval_data)}!")

        # we must split large query file into smaller segments for efficiency
        if len(queries) > max_queries_per_file:
            # split queries into shards because Anserini cannot deal with large query file
            for idx, (qid, query) in enumerate(zip(query_ids, queries)):
                if idx % max_queries_per_file == 0:
                    if idx > 0:
                        g.close()
                    g = open(os.path.join(query_dir, f"queries.{str(idx // max_queries_per_file)}.tsv"), "w")
                g.write("\t".join([str(qid), query]) + "\n")
            g.close()
        else:
            query_path = os.path.join(query_dir, "queries.tsv")
            with open(query_path, "w") as f:
                for qid, qcontent in zip(query_ids, queries):
                    f.write("\t".join([str(qid), qcontent]) + "\n")
        
        query_paths = []
        for query_path in os.listdir(query_dir):
            query_paths.append(os.path.join(query_dir, query_path))
        return query_paths
    
    def _prepare_result(self, result_path):
        retrieval_result = defaultdict(list)
        with open(result_path) as f:
            for line in tqdm(f, desc="Collecting Retrieval Results"):
                fields = line.strip().split("\t")
                qid = int(fields[0])
                tidx = int(fields[1])
                retrieval_result[qid].append(tidx)
        return retrieval_result
    
    def index(self, corpus:Optional[datasets.Dataset]=None, output_dir:str="./bm25", threads:int=32, language:str="en", storeDocvectors:bool=False, load_collection:bool=False, load_index:bool=False, **kwds):
        index_dir = os.path.join(output_dir, "index")
        collection_dir = os.path.join(output_dir, "collection")
        self.output_dir = output_dir
        self.language = language

        if not load_collection and not load_index:
            self._prepare_collection(corpus, collection_dir)            

        if not load_index:
            clear_dir(index_dir)
            args = [
                f"sh {self.anserini_dir}/target/appassembler/bin/IndexCollection -collection JsonCollection -generator DefaultLuceneDocumentGenerator",
                f"-input {collection_dir} -index {index_dir} -threads {threads} -language {language}",
                "-storeDocvectors" if storeDocvectors else ""
            ]
            subprocess.run(" ".join(args), shell=True)
        
    def search(self, eval_data:Union[str, datasets.Dataset], output_dir:Optional[str]=None, k1:Optional[float]=None, b:Optional[float]=None, hits:int=100, threads:int=32, parallelism:int=4, language:Optional[str]=None, max_queries_per_file:int=10000, **kwds):
        if k1 is None:
            k1 = self.k1
        if b is None:
            b = self.b
        
        if output_dir is None and not hasattr(self, "output_dir"):
            raise ValueError(f"Make sure there is an index by either calling .index() or specifying an existing index with index_dir=xxx!")
        elif output_dir is None:
            output_dir = self.output_dir
        if language is None:
            language = self.language

        index_dir = os.path.join(output_dir, "index")
        query_dir = os.path.join(output_dir, "query")

        retrieval_result = {}
        query_paths = self._prepare_query(eval_data, query_dir, max_queries_per_file)

        for path in tqdm(query_paths, desc="Searching"):
            tmp_result_path = path+".tmp"
            args = [
                f"sh {self.anserini_dir}/target/appassembler/bin/SearchCollection -topicreader TsvString -format msmarco",
                f"-index {index_dir} -topics {path} -output {tmp_result_path} -bm25 -bm25.k1 {k1} -bm25.b {b}",
                f"-hits {hits} -threads {threads} -parallelism {parallelism} -language {language}"
            ]
            subprocess.run(" ".join(args), shell=True)
            res = self._prepare_result(tmp_result_path)
            retrieval_result.update(res)
            os.remove(tmp_result_path)

        return list(retrieval_result.keys()), list(retrieval_result.values())


class NaiveBM25Retriever:
    def __init__(self, k1:float=0.9, b:float=0.4, **kwds) -> None:
        self.k1 = k1
        self.b = b

    def index(self, corpus: List[Union[str, List[int]]], verbose: bool=False, stop_tokens: Optional[set]=None):
        """Build in-memory BM25 index."""
        if stop_tokens is None:
            stop_tokens = {}

        dfs = defaultdict(int)
        tfs = []
        inverted_lists = defaultdict(list)
        doc_lengths = np.zeros(len(corpus), dtype=np.float32)

        if verbose:
            iterator = tqdm(corpus, desc="Indexing")
        else:
            iterator = corpus

        for i, doc in enumerate(iterator):
            if isinstance(doc, str):
                doc = doc.split(" ")
                # TODO: stem

            df = {}
            tf = defaultdict(int)
            for token in doc:
                if token not in stop_tokens:
                    tf[token] += 1
                    df[token] = 1
            tfs.append(dict(tf))
            for token in df:
                dfs[token] += 1
                # store the doc offset in the inverted lists of the corresponding token
                inverted_lists[token].append(i)

            doc_lengths[i] = len(doc)

        self.dfs = dict(dfs)
        self.tfs = tfs
        self.doc_length = doc_lengths
        self.inverted_lists = {k: np.array(v) for k, v in inverted_lists.items()}
        self.N = len(corpus)

    def search(self, queries: Union[str, List[int], List[str], List[List[int]]], hits: int=100, k1: Optional[float]=None, b: Optional[float]=None, verbose: bool=False):
        """Search over the BM25 index."""
        if k1 is None:
            k1 = self.k1
        if b is None:
            b = self.b
        
        hits = min(self.N, hits)
        
        global_scores = np.zeros(self.N, dtype=np.float32)
        
        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, list) and isinstance(queries[0], int):
            queries = [queries]
        
        all_scores = np.zeros((len(queries), hits), dtype=np.float32)
        all_indices = np.zeros((len(queries), hits), dtype=np.int64)

        if verbose:
            iterator = tqdm(queries, desc="Searching")
        else:
            iterator = queries
        
        for i, query in enumerate(iterator):
            if isinstance(query, str):
                query = query.split(" ")
                # TODO: stem

            for token in query:
                if token in self.inverted_lists:
                    candidates = self.inverted_lists[token]
                else:
                    continue

                tfs = np.array([self.tfs[candidate][token] for candidate in candidates], dtype=np.float32)
                df = self.dfs[token]
                idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)

                candidate_scores = idf * (k1 + 1) * tfs / (tfs + k1 * (1 - b + b * self.doc_length[candidates]))
                global_scores[candidates] += candidate_scores

            indice = np.argpartition(-global_scores, hits - 1)[:hits]
            score = global_scores[indice]
            
            sorted_idx = np.argsort(score)[::-1]
            indice = indice[sorted_idx]
            score = score[sorted_idx]

            invalid_pos = score == 0
            indice[invalid_pos] = -1
            score[invalid_pos] = -float('inf')

            all_scores[i] = score
            all_indices[i] = indice
        return all_scores, all_indices
