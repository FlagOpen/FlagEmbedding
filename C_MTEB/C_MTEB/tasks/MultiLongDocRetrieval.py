import datasets
from mteb.abstasks import MultilingualTask, AbsTaskRetrieval
from mteb.abstasks.AbsTaskRetrieval import *
# from ...abstasks import MultilingualTask, AbsTaskRetrieval
# from ...abstasks.AbsTaskRetrieval import *


_LANGUAGES = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'ru', 'th', 'zh']


def load_mldr_data(path: str, langs: list, eval_splits: list, cache_dir: str=None):
    corpus = {lang: {split: None for split in eval_splits} for lang in langs}
    queries = {lang: {split: None for split in eval_splits} for lang in langs}
    relevant_docs = {lang: {split: None for split in eval_splits} for lang in langs}
    
    for lang in langs:
        lang_corpus = datasets.load_dataset(path, f'corpus-{lang}', cache_dir=cache_dir)['corpus']
        lang_corpus = {e['docid']: {'text': e['text']} for e in lang_corpus}
        lang_data = datasets.load_dataset(path, lang, cache_dir=cache_dir)
        for split in eval_splits:
            corpus[lang][split] = lang_corpus
            queries[lang][split] = {e['query_id']: e['query'] for e in lang_data[split]}
            relevant_docs[lang][split] = {e['query_id']: {e['positive_passages'][0]['docid']: 1} for e in lang_data[split]}
    
    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)
    return corpus, queries, relevant_docs


class MultiLongDocRetrieval(MultilingualTask, AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MultiLongDocRetrieval',
            'hf_hub_name': 'Shitao/MLDR',
            'reference': 'https://arxiv.org/abs/2402.03216',
            'description': 'MultiLongDocRetrieval: A Multilingual Long-Document Retrieval Dataset',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev', 'test'],
            'eval_langs': _LANGUAGES,
            'main_score': 'ndcg_at_10',
        }
    
    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        
        self.corpus, self.queries, self.relevant_docs = load_mldr_data(
            path=self.description['hf_hub_name'],
            langs=self.langs,
            eval_splits=self.description['eval_splits'],
            cache_dir=kwargs.get('cache_dir', None)
        )
        self.data_loaded = True

    def evaluate(
        self,
        model,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        score_function="cos_sim",
        **kwargs
    ):
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()
        
        model = model if self.is_dres_compatible(model) else DRESModel(model)
        if os.getenv("RANK", None) is None:
            # Non-distributed
            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
            model = DRES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )
        else:
            # Distributed (multi-GPU)
            from beir.retrieval.search.dense import (
                DenseRetrievalParallelExactSearch as DRPES,
            )
            model = DRPES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size,
                **kwargs,
            )
        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        
        scores = {}
        for lang in self.langs:
            print(f"==============================\nStart evaluating {lang} ...")
            corpus, queries, relevant_docs = self.corpus[lang][split], self.queries[lang][split], self.relevant_docs[lang][split]
            
            start_time = time()
            results = retriever.retrieve(corpus, queries)
            end_time = time()
            logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
            ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values, ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
            mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")
            scores[lang] = {
                **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
                **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            }
        
        return scores
