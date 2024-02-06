import datasets
from ...abstasks import MultilingualTask, AbsTaskRetrieval
from ...abstasks.AbsTaskRetrieval import *


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
