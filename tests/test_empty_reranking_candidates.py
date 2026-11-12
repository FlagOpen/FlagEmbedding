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


@pytest.mark.parametrize(
    "results, top_k, ignore_identical_ids",
    [
        ({}, 2, False),
        ({"q1": {}}, 2, False),
        ({"q1": {}, "q2": {}}, 2, False),
        ({"q1": {"q1": 1.0}}, 2, True),
        ({"q1": {"q1": 1.0}, "q2": {"q2": 1.0}}, 2, True),
        ({"q1": {"d1": 1.0}, "q2": {"d1": 1.0}}, 0, False),
    ],
)
def test_empty_candidate_sets_return_empty_scores(
    reranker, results, top_k, ignore_identical_ids
):
    corpus = {docid: {"text": "document one"} for docid in ["q1", "q2", "d1"]}
    queries = {"q1": "query one", "q2": "query two"}

    actual = EvalReranker(reranker, rerank_top_k=top_k)(
        corpus, queries, deepcopy(results), ignore_identical_ids=ignore_identical_ids
    )

    assert actual == {qid: {} for qid in results}


@pytest.mark.parametrize("ignore_identical_ids", [False, True])
def test_empty_queries_do_not_suppress_other_queries_scores(
    reranker, ignore_identical_ids
):
    corpus = {"q1": {"text": "document one"}, "d1": {"text": "document two"}}
    queries = {"q1": "query one", "q2": "query two"}
    results = {"q1": {"q1": 1.0, "d1": 0.5}, "q2": {}}

    actual = EvalReranker(reranker, rerank_top_k=2)(
        corpus, queries, results, ignore_identical_ids=ignore_identical_ids
    )

    assert actual["q2"] == {}
    assert set(actual["q1"]) == ({"d1"} if ignore_identical_ids else {"q1", "d1"})
    assert all(torch.isfinite(torch.tensor(score)) for score in actual["q1"].values())
