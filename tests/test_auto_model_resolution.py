import os

import pytest

from FlagEmbedding.inference.auto_embedder import FlagAutoModel
from FlagEmbedding.inference.auto_reranker import FlagAutoReranker
from FlagEmbedding.inference.embedder.model_mapping import AUTO_EMBEDDER_MAPPING
from FlagEmbedding.inference.reranker.model_mapping import AUTO_RERANKER_MAPPING


class RecordingModel:
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "model_name",
    [
        "jinaai/jina-reranker-v2-base-multilingual",
        "Alibaba-NLP/gte-multilingual-reranker-base",
        "maidalun1020/bce-reranker-base_v1",
        "jinaai/jina-reranker-v1-turbo-en",
    ],
)
def test_auto_reranker_resolves_namespaced_model_ids(monkeypatch, model_name):
    monkeypatch.setattr(
        AUTO_RERANKER_MAPPING[model_name], "model_class", RecordingModel
    )

    model = FlagAutoReranker.from_finetuned(model_name)

    assert model.model_name_or_path == model_name


def test_auto_reranker_resolves_local_model_directory(monkeypatch, tmp_path):
    model_name = "jinaai/jina-reranker-v2-base-multilingual"
    monkeypatch.setattr(
        AUTO_RERANKER_MAPPING[model_name], "model_class", RecordingModel
    )
    model_path = tmp_path / "jina-reranker-v2-base-multilingual"

    model = FlagAutoReranker.from_finetuned(str(model_path))

    assert model.model_name_or_path == str(model_path)


def test_auto_reranker_resolves_local_checkpoint_directory(monkeypatch, tmp_path):
    model_name = "jinaai/jina-reranker-v2-base-multilingual"
    monkeypatch.setattr(
        AUTO_RERANKER_MAPPING[model_name], "model_class", RecordingModel
    )
    checkpoint_path = tmp_path / "jina-reranker-v2-base-multilingual" / "checkpoint-100"

    model = FlagAutoReranker.from_finetuned(str(checkpoint_path))

    assert model.model_name_or_path == str(checkpoint_path)


def test_auto_embedder_resolves_path_with_trailing_separator(monkeypatch, tmp_path):
    model_name = "bge-base-en-v1.5"
    monkeypatch.setattr(
        AUTO_EMBEDDER_MAPPING[model_name], "model_class", RecordingModel
    )
    model_path = tmp_path / model_name
    model_path_with_separator = f"{model_path}{os.sep}"

    model = FlagAutoModel.from_finetuned(model_path_with_separator)

    assert model.model_name_or_path == model_path_with_separator
