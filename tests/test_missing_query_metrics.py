import math

import pytest

pytest.importorskip("faiss")
pytest.importorskip("pytrec_eval")

from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr


@pytest.mark.parametrize("missing_form", ["omitted", "empty", "nonrelevant"])
@pytest.mark.parametrize("cutoff", [1, 2])
def test_missing_retrieval_queries_contribute_zero(missing_form, cutoff):
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    results = {"q1": {"d1": 1.0}}
    if missing_form == "empty":
        results.update({"q2": {}, "q3": {}})
    elif missing_form == "nonrelevant":
        results.update({"q2": {"other": 1.0}, "q3": {"other": 1.0}})

    ndcg, average_precision, recall, precision = evaluate_metrics(
        qrels, results, [cutoff]
    )

    assert ndcg == {f"NDCG@{cutoff}": round(1 / 3, 5)}
    assert average_precision == {f"MAP@{cutoff}": round(1 / 3, 5)}
    assert recall == {f"Recall@{cutoff}": round(1 / 3, 5)}
    assert precision == {f"P@{cutoff}": round(1 / (3 * cutoff), 5)}
    assert evaluate_mrr(qrels, results, [cutoff]) == {f"MRR@{cutoff}": round(1 / 3, 5)}


@pytest.mark.parametrize("results", [{}, {"q1": {}, "q2": {}}])
def test_an_empty_run_has_zero_metrics_for_the_evaluated_queries(results):
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}

    actual = evaluate_metrics(qrels, results, [1, 3])

    assert actual == (
        {"NDCG@1": 0.0, "NDCG@3": 0.0},
        {"MAP@1": 0.0, "MAP@3": 0.0},
        {"Recall@1": 0.0, "Recall@3": 0.0},
        {"P@1": 0.0, "P@3": 0.0},
    )


def test_partial_ranked_hits_are_averaged_over_all_qrels_queries():
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}
    results = {"q1": {"other": 2.0, "d1": 1.0}}

    ndcg, average_precision, recall, precision = evaluate_metrics(qrels, results, [2])

    first_query_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg == {"NDCG@2": round(first_query_ndcg / 2, 5)}
    assert average_precision == {"MAP@2": 0.125}
    assert recall == {"Recall@2": 0.25}
    assert precision == {"P@2": 0.25}


def test_unjudged_queries_do_not_change_the_evaluation_population():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 1.0}, "unjudged": {"other": 3.0}}

    assert evaluate_metrics(qrels, results, [1]) == (
        {"NDCG@1": 0.5},
        {"MAP@1": 0.5},
        {"Recall@1": 0.5},
        {"P@1": 0.5},
    )


def test_a_complete_run_preserves_its_average():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 1.0}, "q2": {"other": 1.0}}

    assert evaluate_metrics(qrels, results, [1]) == (
        {"NDCG@1": 0.5},
        {"MAP@1": 0.5},
        {"Recall@1": 0.5},
        {"P@1": 0.5},
    )
