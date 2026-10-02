"""Native MaxSim regressions: passage padding is excluded from the candidate set."""

import copy
import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import BertNormalizer
from transformers import BertConfig, BertModel, BertTokenizerFast

ROOT = Path(os.environ.get("FLAG_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
EncoderOnlyEmbedderM3Model = importlib.import_module(
    "FlagEmbedding.finetune.embedder.encoder_only.m3.modeling"
).EncoderOnlyEmbedderM3Model


def tiny_model(normalize=True, self_distill=False, kind="m3_kd_loss", sub_batch=-1):
    torch.manual_seed(2)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({f"word{i}": i for i in range(4, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = BertNormalizer()
    tokenizer = BertTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        eos_token="[SEP]",
    )
    base = {
        "model": BertModel(
            BertConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                pad_token_id=0,
            )
        ),
        "colbert_linear": torch.nn.Linear(16, 2),
        "sparse_linear": torch.nn.Linear(16, 1),
    }
    with torch.no_grad():
        base["sparse_linear"].bias.fill_(1.0)
    return EncoderOnlyEmbedderM3Model(
        base,
        tokenizer,
        normalize_embeddings=normalize,
        use_self_distill=self_distill,
        kd_loss_type=kind,
        sub_batch_size=sub_batch,
        temperature=0.7,
    )


def features(ids, padding=0):
    ids = torch.tensor([row + [0] * padding for row in ids])
    return {"input_ids": ids, "attention_mask": ids.ne(0).long()}


def valid_maxsim(q, p, q_mask, p_mask, temperature):
    """Per-pair definition that only visits each passage's real token vectors."""
    rows = []
    for row in range(q.shape[0]):
        columns = []
        query = q[row, q_mask[row, 1:].bool()]
        for column in range(p.shape[0]):
            passage = p[column, p_mask[column, 1:].bool()]
            columns.append((query @ passage.T).max(dim=1).values.mean() / temperature)
        rows.append(torch.stack(columns))
    return torch.stack(rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("padding", [1, 4])
@pytest.mark.parametrize("pad_value", [0.0, 100.0])
def test_passage_mask_matches_valid_token_oracle_and_gradients(
    dtype, padding, pad_value
):
    model = tiny_model()
    q_raw = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]], dtype=dtype, requires_grad=True
    )
    q_mask = torch.tensor([[1, 1, 1, 0]])
    q = q_raw * q_mask[:, 1:, None]
    # The first query token has only negative valid document similarities.
    p = torch.tensor([[[-2.0, 1.0], [-1.0, 3.0]]], dtype=dtype)
    p = torch.cat([p, torch.full((1, padding, 2), pad_value, dtype=dtype)], dim=1)
    p.requires_grad_()
    p_mask = torch.tensor([[1, 1, 1] + [0] * padding])
    expected = valid_maxsim(q, p, q_mask, p_mask, model.temperature)
    actual = model.compute_colbert_score(q, p, q_mask=q_mask, p_mask=p_mask)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (q_raw, p), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (q_raw, p))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
        assert torch.isfinite(actual_grad).all()
    assert actual_grads[0][0, 0].abs().sum() > 0
    assert torch.count_nonzero(actual_grads[1][:, 2:]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_valid_zero_document_vector_remains_a_candidate(dtype):
    model = tiny_model()
    q = torch.tensor([[[1.0, 0.0]]], dtype=dtype)
    p = torch.tensor([[[-2.0, 0.0], [0.0, 0.0], [50.0, 0.0]]], dtype=dtype)
    q_mask, p_mask = torch.tensor([[1, 1]]), torch.tensor([[1, 1, 1, 0]])
    actual = model.compute_colbert_score(q, p, q_mask=q_mask, p_mask=p_mask)
    torch.testing.assert_close(actual, torch.zeros(1, 1, dtype=dtype))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("sub_batch", [-1, 1])
def test_native_encoder_scores_are_independent_of_passage_padding(normalize, sub_batch):
    model = tiny_model(normalize=normalize, sub_batch=sub_batch)
    query = features([[2, 10, 11, 3]])
    passage = features([[2, 5, 3]])
    padded = features([[2, 5, 3]], padding=3)
    qc, pc, padded_pc = (
        model.encode(query)[2],
        model.encode(passage)[2],
        model.encode(padded)[2],
    )
    torch.testing.assert_close(pc, padded_pc[:, : pc.shape[1]])
    expected = valid_maxsim(
        qc, pc, query["attention_mask"], passage["attention_mask"], model.temperature
    )
    actual = model.compute_colbert_score(
        qc, padded_pc, q_mask=query["attention_mask"], p_mask=padded["attention_mask"]
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(
        actual.sum(), model.colbert_linear.weight, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad(expected.sum(), model.colbert_linear.weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("kind", [None, "kl_div", "m3_kd_loss"])
@pytest.mark.parametrize("self_distill", [False, True])
@pytest.mark.parametrize("sub_batch", [-1, 1])
def test_native_unified_training_loss_gradients_and_updates_ignore_passage_padding(
    kind, self_distill, sub_batch
):
    short_model = tiny_model(
        self_distill=self_distill, kind=kind or "m3_kd_loss", sub_batch=sub_batch
    )
    padded_model = copy.deepcopy(short_model)
    query = features([[2, 4, 11, 3], [2, 12, 13, 3]])
    ids = [[2, word, 3] for word in [5, 4, 7, 8]]
    passage, padded = features(ids), features(ids, padding=3)
    teacher_scores = [-0.5, 1.0, 0.8, -0.2] if kind else None
    losses = []
    for model, current_passage in [(short_model, passage), (padded_model, padded)]:
        loss = model(
            queries=query, passages=current_passage, teacher_scores=teacher_scores
        ).loss
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1])
    for name in [
        "model.embeddings.word_embeddings.weight",
        "colbert_linear.weight",
        "sparse_linear.weight",
    ]:
        short_parameter = dict(short_model.named_parameters())[name]
        padded_parameter = dict(padded_model.named_parameters())[name]
        assert short_parameter.grad is not None and short_parameter.grad.abs().sum() > 0
        assert torch.isfinite(short_parameter.grad).all()
        torch.testing.assert_close(
            short_parameter.grad, padded_parameter.grad, atol=1e-6, rtol=1e-5
        )
    before = {name: p.detach().clone() for name, p in short_model.named_parameters()}
    for model in [short_model, padded_model]:
        torch.optim.SGD(model.parameters(), lr=0.01).step()
    for name in [
        "model.embeddings.word_embeddings.weight",
        "colbert_linear.weight",
        "sparse_linear.weight",
    ]:
        short_parameter = dict(short_model.named_parameters())[name]
        padded_parameter = dict(padded_model.named_parameters())[name]
        assert not torch.equal(short_parameter, before[name])
        torch.testing.assert_close(short_parameter, padded_parameter)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("length", [1, 4])
def test_unpadded_positional_api_preserves_existing_scores(dtype, batch, length):
    model = tiny_model()
    torch.manual_seed(13)
    q = torch.randn(batch, length, 3, dtype=dtype)
    p = torch.randn(2 * batch, length, 3, dtype=dtype)
    q_mask = torch.ones(batch, length + 1, dtype=torch.long)
    expected = (
        torch.einsum("qin,pjn->qipj", q, p).max(-1).values.mean(1) / model.temperature
    )
    actual = model.compute_colbert_score(q, p, q_mask)
    torch.testing.assert_close(actual, expected)


def test_list_feature_masks_propagate_to_native_training():
    model = tiny_model(sub_batch=1)
    queries = features([[2, 10, 11, 3], [2, 12, 13, 3]])
    passages = features([[2, word, 3] for word in [5, 6, 7, 8]], padding=3)
    query_list = [
        {key: value[:1] for key, value in queries.items()},
        {key: value[1:] for key, value in queries.items()},
    ]
    passage_list = [
        {key: value[:2] for key, value in passages.items()},
        {key: value[2:] for key, value in passages.items()},
    ]
    expected = copy.deepcopy(model)(
        queries=queries, passages=features([[2, word, 3] for word in [5, 6, 7, 8]])
    ).loss
    actual = model(queries=query_list, passages=passage_list).loss
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(model.colbert_linear.weight.grad).all()
