from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizerFast,
)

pytest.importorskip("faiss")
pytest.importorskip("pytrec_eval")

from FlagEmbedding.abc.evaluation.searcher import EvalDenseRetriever, EvalReranker
from FlagEmbedding.evaluation.bright.searcher import BrightEvalDenseRetriever
from FlagEmbedding.inference.embedder.encoder_only.base import BaseEmbedder
from FlagEmbedding.inference.reranker.encoder_only.base import BaseReranker


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(31)
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
        "fifth",
        "sixth",
        "item",
        "record",
    ]
    encoder_dir = tmp_path_factory.mktemp("filtered_encoder")
    reranker_dir = tmp_path_factory.mktemp("filtered_reranker")
    for path in (encoder_dir, reranker_dir):
        (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        BertTokenizerFast(
            vocab_file=str(path / "vocab.txt"), do_lower_case=False
        ).save_pretrained(path)
    config = BertConfig(
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
    BertModel(config).save_pretrained(encoder_dir)
    BertForSequenceClassification(config).save_pretrained(reranker_dir)
    embedder = BaseEmbedder(
        str(encoder_dir),
        devices=["cpu"],
        use_fp16=False,
        pooling_method="mean",
        batch_size=32,
        query_max_length=16,
        passage_max_length=16,
    )
    reranker = BaseReranker(
        str(reranker_dir), devices=["cpu"], batch_size=32, max_length=16
    )
    return embedder, reranker


@pytest.fixture(scope="module")
def retrieval_data(models):
    corpus = dict(
        zip(
            ["q1", "q2", "d3", "d4", "d5", "d6"],
            [
                {"text": text}
                for text in [
                    "query one",
                    "query two",
                    "document three",
                    "document four",
                    "fifth item",
                    "sixth record",
                ]
            ],
        )
    )
    queries = {"q1": "query one", "q2": "query two", "external": "document three"}
    embedder, _ = models
    # Independent exhaustive NumPy ranking, before any candidate filtering.
    scores = (
        embedder.encode_queries(list(queries.values()))
        @ embedder.encode_corpus([doc["text"] for doc in corpus.values()]).T
    )
    docids = list(corpus)
    rankings = {
        qid: [docids[index] for index in np.argsort(-row)]
        for qid, row in zip(queries, scores)
    }
    assert rankings["q1"][0] == "q1"
    assert rankings["q2"][0] == "q2"
    return corpus, queries, rankings


@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("ignore_identical_ids", [False, True])
def test_dense_retrieval_fills_the_requested_depth_with_eligible_documents(
    models, retrieval_data, top_k, ignore_identical_ids
):
    embedder, _ = models
    corpus, queries, rankings = retrieval_data

    actual = EvalDenseRetriever(embedder, search_top_k=top_k)(
        corpus, queries, ignore_identical_ids=ignore_identical_ids
    )

    for qid in queries:
        expected = [
            docid
            for docid in rankings[qid]
            if not (ignore_identical_ids and docid == qid)
        ][:top_k]
        assert list(actual[qid]) == expected


@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("excluded_count", [0, 1, 3])
@pytest.mark.parametrize("ignore_identical_ids", [False, True])
def test_bright_retrieval_fills_depth_after_qrels_exclusions(
    models, retrieval_data, top_k, excluded_count, ignore_identical_ids
):
    embedder, _ = models
    corpus, queries, rankings = retrieval_data
    qrels = {
        qid: {docid: int(docid not in ranking[:excluded_count]) for docid in corpus}
        for qid, ranking in rankings.items()
    }
    for query_qrels in qrels.values():
        query_qrels["N/A"] = 0

    actual = BrightEvalDenseRetriever(embedder, search_top_k=top_k)(
        corpus,
        queries,
        retriever_qrels=qrels,
        ignore_identical_ids=ignore_identical_ids,
    )

    for qid in queries:
        expected = [
            docid
            for docid in rankings[qid]
            if qrels[qid][docid] == 1 and not (ignore_identical_ids and docid == qid)
        ][:top_k]
        assert list(actual[qid]) == expected


@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("ignore_identical_ids", [False, True])
def test_reranking_selects_the_requested_number_after_identical_id_filtering(
    models, retrieval_data, top_k, ignore_identical_ids
):
    _, reranker = models
    corpus, queries, rankings = retrieval_data
    search_results = {
        qid: {docid: float(len(ranking) - index) for index, docid in enumerate(ranking)}
        for qid, ranking in rankings.items()
    }

    actual = EvalReranker(reranker, rerank_top_k=top_k)(
        corpus,
        queries,
        deepcopy(search_results),
        ignore_identical_ids=ignore_identical_ids,
    )

    for qid in queries:
        expected = [
            docid
            for docid in rankings[qid]
            if not (ignore_identical_ids and docid == qid)
        ][:top_k]
        assert list(actual[qid]) == expected
