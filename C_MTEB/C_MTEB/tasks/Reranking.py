import logging

import numpy as np
from mteb import RerankingEvaluator, AbsTaskReranking
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChineseRerankingEvaluator(RerankingEvaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.
    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.
    :param samples: Must be a list and each element is of the form:
        - {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list of positive
        (relevant) documents, negative is a list of negative (irrelevant) documents.
        - {'query': [], 'positive': [], 'negative': []}. Where query is a list of strings, which embeddings we average
        to get the query embedding.
    """

    def compute_metrics_batched(self, model):
        """
        Computes the metrices in a batched way, by batching all queries and
        all documents together
        """

        if hasattr(model, 'compute_score'):
            return self.compute_metrics_batched_from_crossencoder(model)
        else:
            return self.compute_metrics_batched_from_biencoder(model)

    def compute_metrics_batched_from_crossencoder(self, model):
        all_mrr_scores = []
        all_ap_scores = []

        pairs = []
        for sample in tqdm(self.samples, desc="Evaluating"):
            for p in sample['positive']:
                pairs.append([sample['query'], p])
            for n in sample['negative']:
                pairs.append([sample['query'], n])
        all_scores = model.compute_score(pairs)
        all_scores = np.array(all_scores)

        start_inx = 0
        for sample in tqdm(self.samples, desc="Evaluating"):
            is_relevant = [True] * len(sample['positive']) + [False] * len(sample['negative'])
            pred_scores = all_scores[start_inx:start_inx + len(is_relevant)]
            start_inx += len(is_relevant)

            pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order
            mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
            ap = self.ap_score(is_relevant, pred_scores)

            all_mrr_scores.append(mrr)
            all_ap_scores.append(ap)

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap, "mrr": mean_mrr}

    def compute_metrics_batched_from_biencoder(self, model):
        all_mrr_scores = []
        all_ap_scores = []
        logger.info("Encoding queries...")
        if isinstance(self.samples[0]["query"], str):
            if hasattr(model, 'encode_queries'):
                all_query_embs = model.encode_queries(
                    [sample["query"] for sample in self.samples],
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                )
            else:
                all_query_embs = model.encode(
                    [sample["query"] for sample in self.samples],
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                )
        elif isinstance(self.samples[0]["query"], list):
            # In case the query is a list of strings, we get the most similar embedding to any of the queries
            all_query_flattened = [q for sample in self.samples for q in sample["query"]]
            if hasattr(model, 'encode_queries'):
                all_query_embs = model.encode_queries(all_query_flattened, convert_to_tensor=True,
                                                      batch_size=self.batch_size)
            else:
                all_query_embs = model.encode(all_query_flattened, convert_to_tensor=True, batch_size=self.batch_size)
        else:
            raise ValueError(f"Query must be a string or a list of strings but is {type(self.samples[0]['query'])}")

        logger.info("Encoding candidates...")
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["positive"])
            all_docs.extend(sample["negative"])

        all_docs_embs = model.encode(all_docs, convert_to_tensor=True, batch_size=self.batch_size)

        # Compute scores
        logger.info("Evaluating...")
        query_idx, docs_idx = 0, 0
        for instance in self.samples:
            num_subqueries = len(instance["query"]) if isinstance(instance["query"], list) else 1
            query_emb = all_query_embs[query_idx: query_idx + num_subqueries]
            query_idx += num_subqueries

            num_pos = len(instance["positive"])
            num_neg = len(instance["negative"])
            docs_emb = all_docs_embs[docs_idx: docs_idx + num_pos + num_neg]
            docs_idx += num_pos + num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            is_relevant = [True] * num_pos + [False] * num_neg

            scores = self._compute_metrics_instance(query_emb, docs_emb, is_relevant)
            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap, "mrr": mean_mrr}


def evaluate(self, model, split="test", **kwargs):
    if not self.data_loaded:
        self.load_data()

    data_split = self.dataset[split]

    evaluator = ChineseRerankingEvaluator(data_split, **kwargs)
    scores = evaluator(model)

    return dict(scores)


AbsTaskReranking.evaluate = evaluate


class T2Reranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2Reranking',
            'hf_hub_name': "C-MTEB/T2Reranking",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class T2RerankingZh2En(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2RerankingZh2En',
            'hf_hub_name': "C-MTEB/T2Reranking_zh2en",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh2en'],
            'main_score': 'map',
        }


class T2RerankingEn2Zh(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'T2RerankingEn2Zh',
            'hf_hub_name': "C-MTEB/T2Reranking_en2zh",
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            "reference": "https://arxiv.org/abs/2304.03679",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['en2zh'],
            'main_score': 'map',
        }


class MMarcoReranking(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'MMarcoReranking',
            'hf_hub_name': "C-MTEB/Mmarco-reranking",
            'description': 'mMARCO is a multilingual version of the MS MARCO passage ranking dataset',
            "reference": "https://github.com/unicamp-dl/mMARCO",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class CMedQAv1(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'CMedQAv1',
            "hf_hub_name": "C-MTEB/CMedQAv1-reranking",
            'description': 'Chinese community medical question answering',
            "reference": "https://github.com/zhangsheng93/cMedQA",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }


class CMedQAv2(AbsTaskReranking):
    @property
    def description(self):
        return {
            'name': 'CMedQAv2',
            "hf_hub_name": "C-MTEB/CMedQAv2-reranking",
            'description': 'Chinese community medical question answering',
            "reference": "https://github.com/zhangsheng93/cMedQA2",
            'type': 'Reranking',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }
