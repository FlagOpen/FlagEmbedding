from typing import List, Dict

from mteb.encoder_interface import PromptType
from FlagEmbedding.abc.evaluation.searcher import AbsEmbedder, AbsReranker

class MTEBRetriever(AbsEmbedder):
    def __init__(self, retriever, **kwargs):
        super().__init__(retriever, **kwargs)
    
    def encode_queries(self, queries: List[str], **kwargs):
        return self.retriever.encode_queries(queries, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        return self.retriever.encode_corpus(input_texts, **kwargs)
    
class MTEBReranker(AbsReranker):
    def __init__(self, reranker, **kwargs):
        super().__init__(reranker, **kwargs)