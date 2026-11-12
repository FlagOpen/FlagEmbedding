import json
import os
import random
from collections import Counter

import pytest

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderDataArguments,
    AbsEmbedderSameDatasetTrainDataset,
    AbsEmbedderTrainDataset,
)
from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerDataArguments,
    AbsRerankerTrainDataset,
)


@pytest.fixture(autouse=True)
def preserve_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.fixture
def training_directory(tmp_path):
    directory = tmp_path / "training"
    directory.mkdir()
    for name, count in [("a.jsonl", 8), ("b.jsonl", 12)]:
        rows = [
            {"query": f"{name}:{i}", "pos": [f"positive:{i}"], "neg": [f"negative:{i}"]}
            for i in range(count)
        ]
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "README.txt").write_text("Not a training file.\n")
    return directory


def load_with_directory_order(
    monkeypatch, directory, filenames, dataset_kind, max_examples=100, rank=0
):
    original_listdir = os.listdir

    def listdir(path):
        return (
            filenames
            if os.fspath(path) == os.fspath(directory)
            else original_listdir(path)
        )

    argument_type = (
        AbsRerankerDataArguments
        if dataset_kind == "reranker"
        else AbsEmbedderDataArguments
    )
    args = argument_type(
        train_data=[str(directory)],
        train_group_size=2,
        max_example_num_per_dataset=max_examples,
    )
    if dataset_kind != "reranker":
        args.query_instruction_for_retrieval = ""
    random.seed(123)
    with monkeypatch.context() as patch:
        patch.setattr(os, "listdir", listdir)
        if dataset_kind == "same_dataset":
            return AbsEmbedderSameDatasetTrainDataset(
                args,
                default_batch_size=2,
                seed=123,
                tokenizer=None,
                process_index=rank,
                num_processes=2,
            )
        dataset_type = (
            AbsRerankerTrainDataset
            if dataset_kind == "reranker"
            else AbsEmbedderTrainDataset
        )
        return dataset_type(args, tokenizer=None)


@pytest.mark.parametrize("dataset_kind", ["embedder", "same_dataset", "reranker"])
@pytest.mark.parametrize("max_examples", [6, 100])
def test_directory_order_does_not_change_training_rows(
    monkeypatch, training_directory, dataset_kind, max_examples
):
    forward = load_with_directory_order(
        monkeypatch,
        training_directory,
        ["a.jsonl", "README.txt", "b.jsonl"],
        dataset_kind,
        max_examples,
    )
    reverse = load_with_directory_order(
        monkeypatch,
        training_directory,
        ["b.jsonl", "README.txt", "a.jsonl"],
        dataset_kind,
        max_examples,
    )

    assert list(forward.dataset["query"]) == list(reverse.dataset["query"])
    assert len(forward.dataset) == min(8, max_examples) + min(12, max_examples)


def test_rank_batches_stay_in_one_dataset_and_cover_each_row_once(
    monkeypatch, training_directory
):
    rank_zero = load_with_directory_order(
        monkeypatch, training_directory, ["a.jsonl", "b.jsonl"], "same_dataset", rank=0
    )
    rank_one = load_with_directory_order(
        monkeypatch, training_directory, ["b.jsonl", "a.jsonl"], "same_dataset", rank=1
    )
    all_queries = []
    for batch_index in range(len(rank_zero.batch_datas)):
        queries_zero = rank_zero[batch_index][0]
        queries_one = rank_one[batch_index][0]
        queries = queries_zero + queries_one
        assert len({query.split(":")[0] for query in queries}) == 1
        all_queries.extend(queries)

    expected = list(rank_zero.dataset["query"])
    assert Counter(all_queries) == Counter(expected)
