import torch
import random
import logging
from tqdm import tqdm
from .modeling_dense import DenseRetriever
from .modeling_bm25 import BM25Retriever, NaiveBM25Retriever

logger = logging.getLogger(__name__)


class Retriever:
    """A wrapper for different retrieval_methods."""
    def __init__(self, retrieval_method: str="dense", **kwds) -> None:
        self.retrieval_method = retrieval_method
        self.accelerator = kwds["accelerator"]

        if retrieval_method == "dense":
            self.retriever = DenseRetriever(**kwds)
        elif retrieval_method == "bm25":
            if self.accelerator.process_index == 0:
                self.retriever = BM25Retriever(**kwds)
            else:
                self.retriever = None
        elif retrieval_method == "naive-bm25":
            self.retriever = NaiveBM25Retriever(**kwds)
        else:
            logger.warning(f"Found unimplemented retrieval_method [{retrieval_method}], will return None as query_ids and preds.")
            self.retriever = None

    def to(self, *args, **kwds):
        if hasattr(self.retriever, "to"):
            self.retriever.to(*args, **kwds)
        return self
    
    def encode(self, *args, **kwds):
        if self.retriever is not None and hasattr(self.retriever, "encode"):
            return self.retriever.encode(*args, **kwds)
        else:
            raise NotImplementedError

    def index(self, corpus, **kwds):
        self.corpus_size = len(corpus)
        if self.retriever is not None and hasattr(self.retriever, "index"):
            self.retriever.index(corpus, **kwds)
        self.accelerator.wait_for_everyone()

    def search(self, eval_dataset, **kwds):
        if self.retrieval_method == "dense":
            query_ids = []
            preds = []  # num_samples, hits

            # every process get the same queries while searching different shards
            dataloader = torch.utils.data.DataLoader(
                eval_dataset, 
                batch_size=kwds.get("batch_size", 1000), 
                pin_memory=True,
                num_workers=2,
            )

            for step, inputs in enumerate(tqdm(dataloader, desc="Searching")):
                query_id = inputs.pop("query_id")
                # the indices are already gathered, merged, and sorted inside search function
                score, indice = self.retriever.search(inputs["query"], **kwds)  # batch_size, hits
                query_ids.extend(query_id.tolist())
                preds.extend(indice.tolist())

        elif self.retrieval_method == "bm25" and self.retriever is not None:
            query_ids, preds = self.retriever.search(eval_data=eval_dataset, **kwds)
        
        elif self.retrieval_method == "random":
            query_ids = []
            preds = []
            sample_range = range(self.corpus_size)
            for sample in eval_dataset:
                query_ids.append(sample["query_id"])                
                preds.append(random.sample(sample_range, kwds["hits"]))
                
        elif self.retrieval_method == "naive-bm25":
            raise NotImplementedError(f"Retrieval with naive-bm25 and dataset is not implemented!")

        else:
            query_ids = None
            preds = None

        self.accelerator.wait_for_everyone()
        return query_ids, preds
