from FlagEmbedding.abc.evaluation.utils import evaluate_mrr, evaluate_recall_cap


def test_a_tied_relevant_doc_is_ranked_by_id():
    qrels = {"q": {"b": 1, "a": 0}}
    results = {"q": {"a": 1.0, "b": 1.0}}

    mrr = evaluate_mrr(qrels, results, [1])
    assert mrr["MRR@1"] == 1.0

    capped = evaluate_recall_cap(qrels, results, [1])
    assert capped["R_cap@1"] == 1.0


def test_a_higher_score_still_ranks_first():
    qrels = {"q": {"b": 1}}
    results = {"q": {"a": 0.2, "b": 0.9}}

    mrr = evaluate_mrr(qrels, results, [1])
    assert mrr["MRR@1"] == 1.0
