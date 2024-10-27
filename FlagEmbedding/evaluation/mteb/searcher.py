from typing import List, Dict
from FlagEmbedding.abc.evaluation import EvalDenseRetriever, EvalReranker

class MTEBEvalDenseRetriever(EvalDenseRetriever):
    def __init__(self, embedder, **kwargs):
        super().__init__(embedder, **kwargs)
    
    def set_examples(self, examples_for_task: List[dict] = None):
        self.embedder.set_examples(examples_for_task)

    def set_instruction(self, instruction: str = None):
        self.embedder.query_instruction_for_retrieval = instruction
    
    def get_instruction(self):
        return self.embedder.query_instruction_for_retrieval

    def set_normalize_embeddings(self, normalize_embeddings: bool = True):
        self.embedder.normalize_embeddings = normalize_embeddings
    
    def encode_queries(self, queries: List[str], **kwargs):
        print(kwargs)
        return self.embedder.encode_queries(queries)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        print(kwargs)
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.embedder.encode_corpus(input_texts)
    
    def encode(self, corpus: List[Dict[str, str]], **kwargs):
        print(kwargs)
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.embedder.encode_corpus(input_texts)
    
class MTEBEvalReranker(EvalReranker):
    def __init__(self, reranker, **kwargs):
        super().__init__(reranker, **kwargs)