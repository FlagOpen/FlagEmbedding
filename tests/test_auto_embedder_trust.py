"""Exercise the public dispatcher without loading model or tokenizer code."""

from dataclasses import replace

import pytest

from FlagEmbedding import FlagAutoModel
from FlagEmbedding.inference.embedder.model_mapping import (
    AUTO_EMBEDDER_MAPPING,
    EMBEDDER_CLASS_MAPPING,
    EmbedderModelClass,
)


TRUE_REMOTE_CODE_MODELS = [
    name for name, config in AUTO_EMBEDDER_MAPPING.items() if config.trust_remote_code
]
MODEL_ALIASES = [
    "BAAI/bge-code-v1",
    "another-namespace/bge-code-v1",
    "/shared/models/bge-code-v1",
    "/shared/models/bge-code-v1/checkpoint-100",
]


def capture_constructor(model_class):
    """Replace only the downstream constructor, leaving dispatch unmodified."""
    class Capture:
        DEFAULT_POOLING_METHOD = model_class.DEFAULT_POOLING_METHOD
        selected_model_class = model_class

        def __init__(self, model_name_or_path, **kwargs):
            self.model_name_or_path = model_name_or_path
            self.kwargs = kwargs

    return Capture


@pytest.fixture
def capture_registry(monkeypatch):
    def capture(name):
        config = AUTO_EMBEDDER_MAPPING[name]
        monkeypatch.setitem(
            AUTO_EMBEDDER_MAPPING,
            name,
            replace(config, model_class=capture_constructor(config.model_class)),
        )
        return config

    return capture


@pytest.mark.parametrize(
    "model_name_or_path,registry_name",
    [(f"another-namespace/{name}", name) for name in TRUE_REMOTE_CODE_MODELS]
    + [(path, "bge-code-v1") for path in MODEL_ALIASES],
)
def test_registry_never_implicitly_trusts_custom_code(
    capture_registry, caplog, model_name_or_path, registry_name
):
    config = capture_registry(registry_name)

    model = FlagAutoModel.from_finetuned(model_name_or_path)

    assert model.kwargs["trust_remote_code"] is False
    assert model.selected_model_class is config.model_class
    assert model.model_name_or_path == model_name_or_path
    assert "trust_remote_code=True" in caplog.text
    assert "review" in caplog.text.lower()


@pytest.mark.parametrize("model_name_or_path", MODEL_ALIASES)
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_explicit_trust_is_forwarded(
    capture_registry, model_name_or_path, trust_remote_code
):
    capture_registry("bge-code-v1")

    model = FlagAutoModel.from_finetuned(
        model_name_or_path, trust_remote_code=trust_remote_code
    )

    assert model.kwargs["trust_remote_code"] is trust_remote_code


@pytest.mark.parametrize(
    "name", ["bge-code-v1", "bge-large-en-v1.5", "e5-large-v2"]
)
def test_registry_class_pooling_and_query_format_are_preserved(capture_registry, name):
    config = capture_registry(name)

    model = FlagAutoModel.from_finetuned(f"namespace/{name}")

    assert model.selected_model_class is config.model_class
    assert model.kwargs["pooling_method"] == config.pooling_method.value
    assert model.kwargs["query_instruction_format"] == config.query_instruction_format
    assert model.kwargs["trust_remote_code"] is False


def test_explicit_dispatch_options_are_preserved(capture_registry):
    capture_registry("bge-code-v1")
    options = {
        "trust_remote_code": False,
        "pooling_method": "mean",
        "query_instruction_format": "Query: {}|{}",
        "query_instruction_for_retrieval": "retrieve",
        "normalize_embeddings": False,
        "use_fp16": False,
        "devices": "cpu",
        "truncate_dim": 32,
        "batch_size": 2,
    }

    model = FlagAutoModel.from_finetuned("namespace/bge-code-v1", **options)

    for name, value in options.items():
        assert model.kwargs[name] == value


@pytest.mark.parametrize("trust_remote_code", [None, False, True])
def test_direct_model_class_preserves_explicit_opt_in(monkeypatch, trust_remote_code):
    model_class = EmbedderModelClass.DECODER_ONLY_BASE
    original_class = EMBEDDER_CLASS_MAPPING[model_class]
    monkeypatch.setitem(
        EMBEDDER_CLASS_MAPPING, model_class, capture_constructor(original_class)
    )

    model = FlagAutoModel.from_finetuned(
        "namespace/unregistered-model",
        model_class=model_class.value,
        trust_remote_code=trust_remote_code,
    )

    assert model.kwargs["trust_remote_code"] is (trust_remote_code is True)
    assert model.selected_model_class is original_class
    assert model.kwargs["pooling_method"] == original_class.DEFAULT_POOLING_METHOD
    assert model.kwargs["query_instruction_format"] == "{}{}"


def test_unknown_model_still_requires_a_model_class():
    with pytest.raises(ValueError, match="not found in the model mapping"):
        FlagAutoModel.from_finetuned("namespace/unregistered-model")
