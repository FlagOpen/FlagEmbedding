import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm
from collections import defaultdict



class BM25Retriever:
    def __init__(self, k1:float=0.9, b:float=0.4) -> None:
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
