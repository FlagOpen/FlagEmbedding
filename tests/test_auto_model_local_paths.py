import os

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    PreTrainedTokenizerFast,
)

from FlagEmbedding import FlagAutoModel, FlagAutoReranker


def run_inference(model, kind):
    if kind == "embedder":
        return model.encode(["hello world", "hello"])
    return np.asarray(model.compute_score([("hello", "world"), ("hello", "hello")]))


@pytest.fixture(scope="module", params=["embedder", "reranker"])
def local_model(request, tmp_path_factory):
    """Save small, randomly initialized models without downloading checkpoints."""
    kind = request.param
    name = "bge-base-en-v1.5" if kind == "embedder" else "bge-reranker-base"
    model_dir = tmp_path_factory.mktemp(kind) / name
    checkpoint_dir = model_dir / "checkpoint-10"
    vocabulary = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
        )
    }
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=32,
    )
    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        num_labels=1,
    )
    model_type = BertModel if kind == "embedder" else BertForSequenceClassification
    model = model_type(config)
    for path in (model_dir, checkpoint_dir):
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    factory = FlagAutoModel if kind == "embedder" else FlagAutoReranker
    reference = factory.from_finetuned(str(model_dir), devices="cpu", use_fp16=False)
    expected = run_inference(reference, kind)
    assert np.isfinite(expected).all()
    return kind, factory, model_dir, checkpoint_dir, expected


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("suffix", sorted({"", "/", "//", os.sep, os.sep * 2}))
def test_auto_model_loads_local_directory_with_trailing_separator(
    local_model, checkpoint, suffix
):
    kind, factory, model_dir, checkpoint_dir, expected = local_model
    path = str(checkpoint_dir if checkpoint else model_dir) + suffix

    model = factory.from_finetuned(path, devices="cpu", use_fp16=False)

    assert model.model_name_or_path == path
    np.testing.assert_allclose(
        run_inference(model, kind), expected, rtol=1e-6, atol=1e-6
    )
