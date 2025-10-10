import os
import logging
import gc
import torch
import numpy as np
from typing import Any, Dict, Optional

from FlagEmbedding.abc.evaluation.utils import index, search

from FlagEmbedding.abc.evaluation import EvalRetriever

logger = logging.getLogger(__name__)


class BrightEvalDenseRetriever(EvalRetriever):
    """
    Child class of :class:EvalRetriever for dense retrieval.
    """
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        corpus_embd_save_dir: Optional[str] = None,
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        This is called during the retrieval process.

        Parameters:
            corpus: Dict[str, Dict[str, Any]]: Corpus of documents. 
                Structure: {<docid>: {"text": <text>}}.
                Example: {"doc-0": {"text": "This is a document."}}
            queries: Dict[str, str]: Queries to search for.
                Structure: {<qid>: <query>}.
                Example: {"q-0": "This is a query."}
            corpus_embd_save_dir (Optional[str]): Defaults to :data:`None`.
            ignore_identical_ids (bool): Defaults to :data:`False`.
            **kwargs: Any: Additional arguments.

        Returns: Dict[str, Dict[str, float]]: Top-k search results for each query. k is specified by search_top_k.
            Structure: {qid: {docid: score}}. The higher is the score, the more relevant is the document.
            Example: {"q-0": {"doc-0": 0.9}}
        """
        if ignore_identical_ids:
            logger.warning("ignore_identical_ids is set to True. This means that the search results will not contain identical ids. Note: Dataset such as MIRACL should NOT set this to True.")

        # dense embedding models do not require language as input: AIRBench evaluation
        kwargs.pop("language", None)

        corpus_ids = []
        corpus_texts = []
        for docid, doc in corpus.items():
            corpus_ids.append(docid)
            corpus_texts.append(
                doc["text"] if "title" not in doc 
                else f"{doc['title']} {doc['text']}".strip()
            )
        queries_ids = []
        queries_texts = []
        for qid, query in queries.items():
            queries_ids.append(qid)
            queries_texts.append(query)

        # NOTE: obtain excluded ids from qrels to remove corresponding documents from raw search results
        excluded_ids = {}
        qrels = kwargs.pop("retriever_qrels", None)
        if qrels is not None:
            for qid in qrels:
                excluded_ids[qid] = []
                for docid, score in qrels[qid].items():
                    if score != 1:
                        excluded_ids[qid].append(docid)
        else:
            logger.warning("No qrels provided, so no documents will be excluded.")

        if corpus_embd_save_dir is not None:
            if os.path.exists(os.path.join(corpus_embd_save_dir, "doc.npy")) and not self.overwrite:
                corpus_emb = np.load(os.path.join(corpus_embd_save_dir, "doc.npy"))
            else:
                corpus_emb = self.embedder.encode_corpus(corpus_texts, **kwargs)
        else:
            corpus_emb = self.embedder.encode_corpus(corpus_texts, **kwargs)

        queries_emb = self.embedder.encode_queries(queries_texts, **kwargs)

        # check if the embeddings are in dictionary format: M3Embedder
        if isinstance(corpus_emb, dict):
            corpus_emb = corpus_emb["dense_vecs"]
        if isinstance(queries_emb, dict):
            queries_emb = queries_emb["dense_vecs"]

        if corpus_embd_save_dir is not None and \
            (not os.path.exists(os.path.join(corpus_embd_save_dir, "doc.npy")) or self.overwrite):
            os.makedirs(corpus_embd_save_dir, exist_ok=True)
            np.save(os.path.join(corpus_embd_save_dir, "doc.npy"), corpus_emb)

        gc.collect()
        torch.cuda.empty_cache()

        faiss_index = index(corpus_embeddings=corpus_emb)
        all_scores, all_indices = search(query_embeddings=queries_emb, faiss_index=faiss_index, k=self.search_top_k)

        results = {}
        for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            query_id = queries_ids[idx]

            results[query_id] = {}
            for score, indice in zip(scores, indices):
                if indice != -1:
                    if ignore_identical_ids and corpus_ids[indice] == query_id:
                        continue
                    results[query_id][corpus_ids[indice]] = float(score)

            if qrels is not None:
                # NOTE: Filter out documents with ids in excluded_ids
                for docid in set(excluded_ids[query_id]):
                    if docid != "N/A":
                        results[query_id].pop(docid, None)

            sorted_scores = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)
            # Store the top-k results for the current query
            results[query_id] = {}
            for docid, score in sorted_scores[:self.search_top_k]:
                results[query_id][docid] = float(score)

        return results
