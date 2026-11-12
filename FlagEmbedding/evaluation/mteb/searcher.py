import numpy as np

from typing import List, Dict, Optional
from FlagEmbedding.abc.evaluation import EvalDenseRetriever, EvalReranker


class MTEBEvalDenseRetriever(EvalDenseRetriever):
    """
    Child class of :class:EvalRetriever for MTEB dense retrieval.
    """
    def __init__(self, embedder, **kwargs):
        super().__init__(embedder, **kwargs)
    
    def set_examples(self, examples_for_task: Optional[List[dict]] = None):
        """Set examples for the model.

        Args:
            examples_for_task (Optional[List[dict]], optional): Examples for the task. Defaults to None.
        """
        self.embedder.set_examples(examples_for_task)

    def set_instruction(self, instruction: Optional[str] = None):
        """Set the instruction to use for the embedding model.

        Args:
            instruction (Optional[str], optional): _description_. Defaults to None.
        """
        self.embedder.query_instruction_for_retrieval = instruction
    
    def get_instruction(self):
        """Get the instruction of embedding model.

        Returns:
            str: Instruction
        """
        return self.embedder.query_instruction_for_retrieval

    def set_normalize_embeddings(self, normalize_embeddings: bool = True):
        """Set whether normalize the output embeddings

        Args:
            normalize_embeddings (bool, optional): Boolean to control whether or not normalize the embeddings. Defaults to ``True``.
        """
        self.embedder.normalize_embeddings = normalize_embeddings
    
    def stop_pool(self):
        self.embedder.stop_self_pool()
        try:
            self.embedder.stop_self_query_pool()
        except:
            pass

    def encode_queries(self, queries: List[str], **kwargs):
        """Encode input queries.

        Args:
            queries (List[str]): Input queries.

        Returns:
            Union[np.ndarray, torch.Tensor]: Query embeddings.
        """
        emb = self.embedder.encode_queries(queries)
        if isinstance(emb, dict):
            emb = emb["dense_vecs"]
        return emb.astype(np.float32)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        """Encode input corpus.

        Args:
            corpus (List[Dict[str, str]]): Input corpus.

        Returns:
            Union[np.ndarray, torch.Tensor]: Corpus embeddings.
        """
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        emb = self.embedder.encode_corpus(input_texts)
        if isinstance(emb, dict):
            emb = emb["dense_vecs"]
        return emb.astype(np.float32)
    
    def encode(self, corpus: List[Dict[str, str]], **kwargs):
        """Encode the imput.

        Args:
            corpus (List[Dict[str, str]]): Input corpus or sentences.

        Returns:
            Union[np.ndarray, torch.Tensor]: Corpus embeddings.
        """
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        emb = self.embedder.encode_queries(input_texts)
        if isinstance(emb, dict):
            emb = emb["dense_vecs"]
        return emb.astype(np.float32)

class MTEBEvalReranker(EvalReranker):
    """
    Child class of :class:EvalReranker for reranker in MTEB.
    """
    def __init__(self, reranker, **kwargs):
        super().__init__(reranker, **kwargs)
