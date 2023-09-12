from collections import defaultdict

from datasets import load_dataset, DatasetDict
from mteb import AbsTaskRetrieval


def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(hf_hub_name)
    qrels = load_dataset(hf_hub_name + '-qrels')[eval_split]

    corpus = {e['id']: {'text': e['text']} for e in dataset['corpus']}
    queries = {e['id']: e['text'] for e in dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['qid']][e['pid']] = e['score']

    corpus = DatasetDict({eval_split:corpus})
    queries = DatasetDict({eval_split:queries})
    relevant_docs = DatasetDict({eval_split:relevant_docs})
    return corpus, queries, relevant_docs


class T2Retrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'T2Retrieval',
            'hf_hub_name': 'C-MTEB/T2Retrieval',
            'reference': 'https://arxiv.org/abs/2304.03679',
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class MMarcoRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MMarcoRetrieval',
            'hf_hub_name': 'C-MTEB/MMarcoRetrieval',
            'reference': 'https://github.com/unicamp-dl/mMARCO',
            'description': 'mMARCO is a multilingual version of the MS MARCO passage ranking dataset',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class DuRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'DuRetrieval',
            'hf_hub_name': 'C-MTEB/DuRetrieval',
            'reference': 'https://aclanthology.org/2022.emnlp-main.357.pdf',
            'description': 'A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class CovidRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CovidRetrieval',
            'hf_hub_name': 'C-MTEB/CovidRetrieval',
            'reference': 'https://aclanthology.org/2022.emnlp-main.357.pdf',
            'description': 'COVID-19 news articles',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True



class CmedqaRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CmedqaRetrieval',
            'hf_hub_name': 'C-MTEB/CmedqaRetrieval',
            'reference': 'https://aclanthology.org/2022.emnlp-main.357.pdf',
            'description': 'Online medical consultation text',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class EcomRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'EcomRetrieval',
            'hf_hub_name': 'C-MTEB/EcomRetrieval',
            'reference': 'https://arxiv.org/abs/2203.03367',
            'description': 'Passage retrieval dataset collected from Alibaba search engine systems in ecom domain',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class MedicalRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MedicalRetrieval',
            'hf_hub_name': 'C-MTEB/MedicalRetrieval',
            'reference': 'https://arxiv.org/abs/2203.03367',
            'description': 'Passage retrieval dataset collected from Alibaba search engine systems in medical domain',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class VideoRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'VideoRetrieval',
            'hf_hub_name': 'C-MTEB/VideoRetrieval',
            'reference': 'https://arxiv.org/abs/2203.03367',
            'description': 'Passage retrieval dataset collected from Alibaba search engine systems in video domain',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'], self.description['eval_splits'])
        self.data_loaded = True

