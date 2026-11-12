from copy import deepcopy

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)

pytest.importorskip("faiss")
pytest.importorskip("pytrec_eval")

from FlagEmbedding.abc.evaluation.searcher import EvalReranker
from FlagEmbedding.inference.reranker.encoder_only.base import BaseReranker


@pytest.fixture(scope="module")
def reranker(tmp_path_factory):
    torch.manual_seed(7)
    path = tmp_path_factory.mktemp("native_reranker")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "query",
        "document",
        "one",
        "two",
        "three",
        "four",
    ]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(
        vocab_file=str(path / "vocab.txt"), do_lower_case=False
    )
    tokenizer.save_pretrained(path)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_labels=1,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout=0.0,
            pad_token_id=0,
        )
    ).save_pretrained(path)
    return BaseReranker(str(path), devices=["cpu"], batch_size=16, max_length=16)


def retrieval_inputs():
    corpus = {
        name: {"text": "document " + word}
        for name, word in zip(["q1", "d2", "d3", "d4"], ["one", "two", "three", "four"])
    }
    queries = {"q1": "query one", "q2": "query two"}
    results = {
        "q1": {"d2": 0.3, "q1": 0.9, "d4": -0.2, "d3": 0.6},
        "q2": {"d3": 0.1, "d4": 0.8, "q1": 0.2, "d2": 0.7},
    }
    return corpus, queries, results


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("ignore_identical_ids", [False, True])
def test_reranking_preserves_the_full_retrieval_input(
    reranker, top_k, ignore_identical_ids
):
    corpus, queries, results = retrieval_inputs()
    original = deepcopy(results)
    query_mappings = dict(results)

    actual = EvalReranker(reranker, rerank_top_k=top_k)(
        corpus, queries, results, ignore_identical_ids=ignore_identical_ids
    )

    assert results == original
    assert all(results[qid] is query_mappings[qid] for qid in results)
    for qid, scores in original.items():
        selected = sorted(scores, key=scores.get, reverse=True)[:top_k]
        expected_ids = {
            docid for docid in selected if not (ignore_identical_ids and qid == docid)
        }
        assert set(actual[qid]) == expected_ids


@pytest.mark.parametrize("first_k, second_k", [(1, 3), (3, 1)])
def test_comparing_reranking_depths_does_not_lose_candidates(
    reranker, first_k, second_k
):
    corpus, queries, results = retrieval_inputs()
    fresh_results = deepcopy(results)
    EvalReranker(reranker, rerank_top_k=first_k)(corpus, queries, results)

    actual = EvalReranker(reranker, rerank_top_k=second_k)(corpus, queries, results)
    expected = EvalReranker(reranker, rerank_top_k=second_k)(
        corpus, queries, fresh_results
    )

    assert actual == expected
    assert all(len(scores) == second_k for scores in actual.values())
