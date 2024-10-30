from typing import List, Dict, Optional
from FlagEmbedding.abc.evaluation import EvalDenseRetriever, EvalReranker


class MTEBEvalDenseRetriever(EvalDenseRetriever):
    def __init__(self, embedder, **kwargs):
        super().__init__(embedder, **kwargs)
    
    def set_examples(self, examples_for_task: Optional[List[dict]] = None):
        self.embedder.set_examples(examples_for_task)

    def set_instruction(self, instruction: Optional[str] = None):
        self.embedder.query_instruction_for_retrieval = instruction
    
    def get_instruction(self):
        return self.embedder.query_instruction_for_retrieval

    def set_normalize_embeddings(self, normalize_embeddings: bool = True):
        self.embedder.normalize_embeddings = normalize_embeddings
    
    def encode_queries(self, queries: List[str], **kwargs):
        emb = self.embedder.encode_queries(queries)
        if isinstance(emb, dict):
            emb = emb["dense_vecs"]
        return emb
    
    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        emb = self.embedder.encode_corpus(input_texts)
        if isinstance(emb, dict):
            emb = emb["dense_vecs"]
        return emb
    
    def encode(self, corpus: List[Dict[str, str]], **kwargs):
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        emb = self.embedder.encode_queries(input_texts)
        if isinstance(emb, dict):
            emb = emb["dense_vecs"]
        return emb

class MTEBEvalReranker(EvalReranker):
    def __init__(self, reranker, **kwargs):
        super().__init__(reranker, **kwargs)
