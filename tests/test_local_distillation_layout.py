"""Native CPU regressions for per-query negative groups and M3 distillation."""

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
AbsEmbedderModel = importlib.import_module(
    "FlagEmbedding.abc.finetune.embedder.AbsModeling"
).AbsEmbedderModel
BiEncoderOnlyEmbedderModel = importlib.import_module(
    "FlagEmbedding.finetune.embedder.encoder_only.base.modeling"
).BiEncoderOnlyEmbedderModel
EncoderOnlyEmbedderM3Model = importlib.import_module(
    "FlagEmbedding.finetune.embedder.encoder_only.m3.modeling"
).EncoderOnlyEmbedderM3Model


def sequential_oracle(scores, targets, group_size, local):
    """Per-query weighted conditional log likelihood with selected columns removed."""
    losses = []
    for row in range(scores.shape[0]):
        remaining = list(range(scores.shape[1]))
        loss = scores[row].sum() * 0
        start = 0 if local else row * group_size
        for offset in range(group_size):
            column = start + offset
            log_partition = torch.logsumexp(scores[row, remaining], dim=0)
            loss = loss + targets[row, offset] * (log_partition - scores[row, column])
            remaining.remove(column)
        losses.append(loss)
    return torch.stack(losses).mean()


def local_columns(matrix, batch, group):
    return torch.stack(
        [matrix[row, row * group : (row + 1) * group] for row in range(batch)]
    )


def component_loss(scores, teacher, kind):
    if teacher is None:
        return (torch.logsumexp(scores, dim=1) - scores[:, 0]).mean()
    if kind == "m3_kd_loss":
        return sequential_oracle(scores, teacher, scores.shape[1], local=True)
    return (
        -(teacher * scores.log_softmax(-1)).sum(-1).mean()
        + (torch.logsumexp(scores, dim=1) - scores[:, 0]).mean()
    )


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("group", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("local", [False, True])
def test_m3_loss_and_score_gradients_match_conditional_likelihood(
    batch, group, dtype, local
):
    torch.manual_seed(7)
    scores = torch.randn(
        batch, group if local else batch * group, dtype=dtype, requires_grad=True
    )
    teacher = torch.softmax(torch.randn(batch, group, dtype=dtype), dim=-1)
    expected = sequential_oracle(scores, teacher, group, local)
    actual = AbsEmbedderModel.distill_loss(
        "m3_kd_loss", teacher, scores, group_size=group
    )
    torch.testing.assert_close(actual, expected)
    (actual_grad,) = torch.autograd.grad(actual, scores, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, scores)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()


def tiny_components():
    torch.manual_seed(19)
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
    model = BertModel(
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
    )
    sparse = torch.nn.Linear(16, 1)
    with torch.no_grad():
        sparse.bias.fill_(1.0)
    return tokenizer, {
        "model": model,
        "colbert_linear": torch.nn.Linear(16, 8),
        "sparse_linear": sparse,
    }


def features(count, offset=0):
    ids = torch.tensor([[2, 4, 5 + (index + offset) % 25, 3] for index in range(count)])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("kind", ["kl_div", "m3_kd_loss"])
@pytest.mark.parametrize("batch,group", [(2, 3), (4, 2)])
def test_native_dense_training_handles_local_teacher_groups(kind, batch, group):
    tokenizer, base = tiny_components()
    model = BiEncoderOnlyEmbedderModel(
        base["model"], tokenizer, kd_loss_type=kind, temperature=0.7
    )
    queries, passages = features(batch), features(batch * group, 5)
    teacher_scores = torch.linspace(-0.7, 1.5, batch * group).view(batch, group)
    teacher = teacher_scores.softmax(-1)
    query_embeddings, passage_embeddings = model.encode(queries), model.encode(passages)
    local = local_columns(query_embeddings @ passage_embeddings.T / 0.7, batch, group)
    expected = component_loss(local, teacher, kind)
    actual = model(
        queries=queries,
        passages=passages,
        teacher_scores=teacher_scores.flatten().tolist(),
        no_in_batch_neg_flag=True,
    ).loss
    torch.testing.assert_close(actual, expected)
    actual.backward()
    gradient = base["model"].embeddings.word_embeddings.weight.grad
    assert (
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
    )


@pytest.mark.parametrize("kind", [None, "kl_div", "m3_kd_loss"])
@pytest.mark.parametrize("batch,group", [(2, 3), (4, 2)])
@pytest.mark.parametrize("self_distill", [False, True])
def test_native_unified_m3_local_objective_and_updates(
    kind, batch, group, self_distill
):
    tokenizer, base = tiny_components()
    model = EncoderOnlyEmbedderM3Model(
        base,
        tokenizer,
        kd_loss_type=kind or "m3_kd_loss",
        temperature=0.7,
        unified_finetuning=True,
        use_self_distill=self_distill,
    )
    queries, passages = features(batch), features(batch * group, 5)
    teacher_scores = (
        torch.linspace(-0.7, 1.5, batch * group).view(batch, group) if kind else None
    )
    teacher = teacher_scores.softmax(-1) if kind else None
    q_dense, q_sparse, q_colbert = model.encode(queries)
    p_dense, p_sparse, p_colbert = model.encode(passages)
    dense = local_columns(q_dense @ p_dense.T / 0.7, batch, group)
    sparse = local_columns(q_sparse @ p_sparse.T / 0.7, batch, group)
    token_scores = torch.einsum("qid,pjd->qipj", q_colbert, p_colbert)
    colbert = local_columns(token_scores.max(-1).values.sum(1) / 3 / 0.7, batch, group)
    ensemble = dense + 0.3 * sparse + colbert
    expected = (
        component_loss(dense, teacher, kind)
        + component_loss(ensemble, teacher, kind)
        + 0.1 * component_loss(sparse, teacher, kind)
        + component_loss(colbert, teacher, kind)
    ) / 4
    if self_distill:
        own_teacher = ensemble.detach().softmax(-1)

        def soft_ce(scores):
            return -(own_teacher * scores.log_softmax(-1)).sum(-1).mean()

        expected = (
            expected + (soft_ce(dense) + 0.1 * soft_ce(sparse) + soft_ce(colbert)) / 3
        ) / 2
    actual = model(
        queries=queries,
        passages=passages,
        teacher_scores=teacher_scores.flatten().tolist() if kind else None,
        no_in_batch_neg_flag=True,
    ).loss
    torch.testing.assert_close(actual, expected)
    assert model.step == 1
    before = [
        parameter.detach().clone()
        for parameter in [
            base["model"].embeddings.word_embeddings.weight,
            base["colbert_linear"].weight,
            base["sparse_linear"].weight,
        ]
    ]
    actual.backward()
    parameters = [
        base["model"].embeddings.word_embeddings.weight,
        base["colbert_linear"].weight,
        base["sparse_linear"].weight,
    ]
    for parameter in parameters:
        assert (
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert all(
        not torch.equal(parameter, initial)
        for parameter, initial in zip(parameters, before)
    )


def test_local_score_preserves_each_queries_ensemble_group():
    tokenizer, base = tiny_components()
    model = BiEncoderOnlyEmbedderModel(base["model"], tokenizer)
    queries, passages = torch.randn(4, 16), torch.randn(8, 16)
    local_scores = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    actual = model.compute_local_score(
        queries, passages, compute_score_func=lambda q, p: local_scores
    )
    torch.testing.assert_close(actual, local_scores)


@pytest.mark.parametrize("kind", [None, "kl_div", "m3_kd_loss"])
@pytest.mark.parametrize("self_distill", [False, True])
def test_native_unified_in_batch_objective_remains_unchanged(kind, self_distill):
    tokenizer, base = tiny_components()
    batch, group = 2, 3
    model = EncoderOnlyEmbedderM3Model(
        base,
        tokenizer,
        kd_loss_type=kind or "m3_kd_loss",
        temperature=0.7,
        unified_finetuning=True,
        use_self_distill=self_distill,
    )
    queries, passages = features(batch), features(batch * group, 5)
    teacher_scores = (
        torch.linspace(-0.7, 1.5, batch * group).view(batch, group) if kind else None
    )
    teacher = teacher_scores.softmax(-1) if kind else None
    q_dense, q_sparse, q_colbert = model.encode(queries)
    p_dense, p_sparse, p_colbert = model.encode(passages)
    dense = q_dense @ p_dense.T / 0.7
    sparse = q_sparse @ p_sparse.T / 0.7
    colbert = (
        torch.einsum("qid,pjd->qipj", q_colbert, p_colbert).max(-1).values.sum(1)
        / 3
        / 0.7
    )
    ensemble = dense + 0.3 * sparse + colbert

    def expected_component(scores):
        if kind == "m3_kd_loss":
            return sequential_oracle(scores, teacher, group, local=False)
        positive = scores[torch.arange(batch), torch.arange(batch) * group]
        loss = (torch.logsumexp(scores, dim=1) - positive).mean()
        if teacher is not None:
            local = local_columns(scores, batch, group)
            loss = loss - (teacher * local.log_softmax(-1)).sum(-1).mean()
        return loss

    expected = (
        expected_component(dense)
        + expected_component(ensemble)
        + 0.1 * expected_component(sparse)
        + expected_component(colbert)
    ) / 4
    if self_distill:
        own_teacher = ensemble.detach().softmax(-1)

        def soft_ce(scores):
            return -(own_teacher * scores.log_softmax(-1)).sum(-1).mean()

        expected = (
            expected + (soft_ce(dense) + 0.1 * soft_ce(sparse) + soft_ce(colbert)) / 3
        ) / 2
    actual = model(
        queries=queries,
        passages=passages,
        teacher_scores=teacher_scores.flatten().tolist() if kind else None,
        no_in_batch_neg_flag=False,
    ).loss
    torch.testing.assert_close(actual, expected)
