import copy

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from FlagEmbedding.abc.finetune.embedder.AbsDataset import (
    AbsEmbedderCollator,
    AbsEmbedderSameDatasetCollator,
)
from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
    EncoderOnlyEmbedderM3Model,
)


@pytest.fixture
def tokenizer(tmp_path):
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefgh")
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    tokenizer.eos_token = tokenizer.sep_token
    return tokenizer


def make_model(tokenizer, unified=True):
    torch.manual_seed(37)
    backbone = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=12,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    return EncoderOnlyEmbedderM3Model(
        {
            "model": backbone,
            "colbert_linear": torch.nn.Linear(12, 4),
            "sparse_linear": torch.nn.Linear(12, 1),
        },
        tokenizer=tokenizer,
        unified_finetuning=unified,
        normalize_embeddings=True,
        temperature=0.7,
    ).double()


def collate(tokenizer, kind, sub_batch_size, with_teacher=False):
    queries = ["a b c d e", "b c d", "f"]
    passages = ["a b c d", "c b", "b c d e f", "e", "f g", "h"]
    scores = [3.0, -2.0, 2.0, -1.0, 1.0, -3.0] if with_teacher else None
    options = {"tokenizer": tokenizer, "sub_batch_size": sub_batch_size}
    if kind == "same-dataset":
        return AbsEmbedderSameDatasetCollator(**options)(
            [(queries, passages, scores, False)]
        )
    rows = [
        (
            query,
            passages[2 * row : 2 * row + 2],
            scores[2 * row : 2 * row + 2] if scores is not None else None,
        )
        for row, query in enumerate(queries)
    ]
    return AbsEmbedderCollator(**options)(rows)


@pytest.mark.parametrize("kind", ["standard", "same-dataset"])
@pytest.mark.parametrize("sub_batch_size", [1, 2])
@pytest.mark.parametrize("unified", [True, False])
def test_dynamic_subbatches_match_full_batch_embeddings(
    tokenizer, kind, sub_batch_size, unified
):
    model = make_model(tokenizer, unified)
    split = collate(tokenizer, kind, sub_batch_size)
    full = collate(tokenizer, kind, -1)

    for field in ("queries", "passages"):
        assert len({part["input_ids"].shape[1] for part in split[field]}) > 1
        actual = model.encode(split[field])
        expected = model.encode(full[field])
        for actual_vectors, expected_vectors in zip(actual, expected):
            if expected_vectors is None:
                assert actual_vectors is None
            else:
                torch.testing.assert_close(actual_vectors, expected_vectors)


@pytest.mark.parametrize("kind", ["standard", "same-dataset"])
@pytest.mark.parametrize("with_teacher", [False, True])
@pytest.mark.parametrize("self_distill", [False, True])
def test_dynamic_subbatch_training_matches_full_batch_gradients(
    tokenizer, kind, with_teacher, self_distill
):
    model = make_model(tokenizer).train()
    model.use_self_distill = self_distill
    reference = copy.deepcopy(model)
    split = collate(tokenizer, kind, 2, with_teacher)
    full = collate(tokenizer, kind, -1, with_teacher)

    loss = model(**split).loss
    expected_loss = reference(**full).loss
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    expected_loss.backward()
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        if expected_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, rtol=1e-7, atol=1e-10
            )


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_subbatch_padding_preserves_existing_vectors_and_mask_alignment(
    tokenizer, padding_side
):
    tokenizer.padding_side = padding_side
    model = make_model(tokenizer)
    features = collate(tokenizer, "standard", 2)["queries"]
    separate = [model.encode(part)[2] for part in features]

    combined = model.encode(features)[2]
    mask = model._get_queries_attention_mask(features)[:, 1:]

    assert combined.shape[:2] == mask.shape
    offset = 0
    for vectors, part in zip(separate, features):
        rows = combined[offset : offset + vectors.shape[0]]
        mask_rows = mask[offset : offset + vectors.shape[0]]
        length = vectors.shape[1]
        valid_slice = rows[:, :length] if padding_side == "right" else rows[:, -length:]
        torch.testing.assert_close(valid_slice, vectors)
        padding = rows[:, length:] if padding_side == "right" else rows[:, :-length]
        assert torch.count_nonzero(padding) == 0
        mask_slice = mask_rows[:, :length] if padding_side == "right" else mask_rows[:, -length:]
        torch.testing.assert_close(mask_slice, part["attention_mask"][:, 1:])
        mask_padding = mask_rows[:, length:] if padding_side == "right" else mask_rows[:, :-length]
        assert torch.count_nonzero(mask_padding) == 0
        offset += vectors.shape[0]


@pytest.mark.parametrize("sub_batch_size", [1, 2])
def test_tensor_subbatch_control_matches_unsplit_encoding(tokenizer, sub_batch_size):
    model = make_model(tokenizer)
    features = collate(tokenizer, "standard", -1)["queries"]
    expected = model.encode(features)
    model.sub_batch_size = sub_batch_size

    actual = model.encode(features)

    for actual_vectors, expected_vectors in zip(actual, expected):
        torch.testing.assert_close(actual_vectors, expected_vectors)
