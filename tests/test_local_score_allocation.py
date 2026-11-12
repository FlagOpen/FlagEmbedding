import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from FlagEmbedding.abc.finetune.embedder.AbsModeling import AbsEmbedderModel
from FlagEmbedding.finetune.embedder.decoder_only.base.modeling import (
    BiDecoderOnlyEmbedderModel,
)
from FlagEmbedding.finetune.embedder.decoder_only.icl.modeling import (
    BiDecoderOnlyEmbedderICLModel,
)
from FlagEmbedding.finetune.embedder.encoder_only.base.modeling import (
    BiEncoderOnlyEmbedderModel,
)


def make_model(kind, use_mrl=False):
    torch.manual_seed(14)
    if kind == "encoder":
        backbone = BertModel(
            BertConfig(
                vocab_size=24,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        )
        model_type = BiEncoderOnlyEmbedderModel
    else:
        backbone = GPT2Model(
            GPT2Config(
                vocab_size=24,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        model_type = (
            BiDecoderOnlyEmbedderModel
            if kind == "decoder"
            else BiDecoderOnlyEmbedderICLModel
        )
    return model_type(
        backbone, temperature=0.7, use_mrl=use_mrl, mrl_dims=[4, 8] if use_mrl else []
    )


def per_query_scores(queries, passages):
    group_size = passages.shape[0] // queries.shape[0]
    return (
        torch.stack(
            [
                torch.stack(
                    [
                        torch.dot(query, passages[row * group_size + column])
                        for column in range(group_size)
                    ]
                )
                for row, query in enumerate(queries)
            ]
        )
        / 0.7
    )


@pytest.mark.parametrize("kind", ["encoder", "decoder", "icl"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_local_scores_and_gradients_match_per_query_dot_products(
    kind, dtype, noncontiguous
):
    model = make_model(kind)
    queries = (
        torch.randn(3, 16, 2, dtype=dtype)[..., 0]
        if noncontiguous
        else torch.randn(3, 16, dtype=dtype)
    )
    passages = (
        torch.randn(9, 16, 2, dtype=dtype)[..., 0]
        if noncontiguous
        else torch.randn(9, 16, dtype=dtype)
    )
    queries.requires_grad_()
    passages.requires_grad_()

    actual = model.compute_local_score(queries, passages)
    expected = per_query_scores(queries, passages)

    torch.testing.assert_close(actual, expected)
    weights = torch.arange(1, 10, dtype=dtype).reshape(3, 3)
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(), (queries, passages), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(), (queries, passages)
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("kind", ["encoder", "decoder", "icl"])
@pytest.mark.parametrize("use_mrl", [False, True])
@pytest.mark.parametrize("with_teacher", [False, True])
def test_native_local_negative_training_preserves_loss_and_gradients(
    kind, use_mrl, with_teacher
):
    model = make_model(kind, use_mrl).double().train()
    query_ids = torch.tensor([[1, 2, 3], [1, 4, 5]])
    passage_ids = torch.tensor(
        [[1, 3, 4], [1, 3, 6], [1, 3, 7], [1, 5, 8], [1, 5, 9], [1, 5, 10]]
    )
    queries = {"input_ids": query_ids, "attention_mask": torch.ones_like(query_ids)}
    passages = {
        "input_ids": passage_ids,
        "attention_mask": torch.ones_like(passage_ids),
    }
    teacher = [1.0, -1.0, 0.5, 2.0, -0.5, 0.0] if with_teacher else None

    actual = model(
        queries=queries,
        passages=passages,
        teacher_scores=teacher,
        no_in_batch_neg_flag=True,
    ).loss

    query_embeddings = model.encode(queries)
    passage_embeddings = model.encode(passages)
    if not use_mrl:
        query_embeddings, passage_embeddings = [query_embeddings], [passage_embeddings]
    losses = []
    for query_vectors, passage_vectors in zip(query_embeddings, passage_embeddings):
        scores = per_query_scores(query_vectors, passage_vectors)
        loss = F.cross_entropy(scores, torch.zeros(2, dtype=torch.long))
        if with_teacher:
            probabilities = torch.tensor(teacher).reshape(2, 3).softmax(-1)
            loss -= (scores.log_softmax(-1) * probabilities).sum(-1).mean()
        losses.append(loss)
    expected = sum(losses) / len(losses)
    # The existing MRL forward accumulates its losses in float32.
    torch.testing.assert_close(actual, expected.float() if use_mrl else expected)
    parameters = list(model.parameters())
    actual_gradients = torch.autograd.grad(
        actual, parameters, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-7, atol=1e-10
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_scoring_does_not_allocate_cross_query_scores(dtype):
    model = make_model("encoder")
    batch_size, group_size = 128, 8
    queries = torch.randn(batch_size, 16, dtype=dtype)
    passages = torch.randn(batch_size * group_size, 16, dtype=dtype)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as recorded:
        actual = model.compute_local_score(queries, passages)

    assert actual.shape == (batch_size, group_size)
    largest_allocation = max(event.cpu_memory_usage for event in recorded.events())
    # Allow several local-sized buffers, but not a batch x (batch * group) matrix.
    assert largest_allocation <= 4 * actual.numel() * actual.element_size()


def test_custom_score_callbacks_keep_their_full_matrix_contract():
    model = make_model("encoder")
    queries, passages = torch.randn(2, 16), torch.randn(6, 16)

    def squared_scores(q, p, offset):
        return (q @ p.T + offset).square()

    actual = model.compute_local_score(
        queries, passages, compute_score_func=squared_scores, offset=0.2
    )
    full = squared_scores(queries, passages, 0.2)
    torch.testing.assert_close(actual, torch.stack([full[0, :3], full[1, 3:]]))


def test_custom_embedder_keeps_the_two_dimensional_scoring_contract():
    class TwoDimensionalEmbedder(AbsEmbedderModel):
        def encode(self, features):
            return self.model(features)

        def compute_score(self, q_reps, p_reps):
            assert q_reps.ndim == p_reps.ndim == 2
            return q_reps @ p_reps.T

        def compute_loss(self, scores, target):
            return F.cross_entropy(scores, target)

        def save(self, output_dir):
            torch.save(self.state_dict(), output_dir)

    model = TwoDimensionalEmbedder(torch.nn.Linear(4, 3)).train()
    queries, passages = torch.randn(2, 4), torch.randn(6, 4)

    actual = model(queries=queries, passages=passages, no_in_batch_neg_flag=True).loss

    full_scores = model.encode(queries) @ model.encode(passages).T
    expected = F.cross_entropy(
        torch.stack([full_scores[0, :3], full_scores[1, 3:]]),
        torch.zeros(2, dtype=torch.long),
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert model.model.weight.grad is not None


@pytest.mark.parametrize("kind", ["encoder", "decoder", "icl"])
def test_dense_subclasses_keep_custom_two_dimensional_scores(kind):
    original = make_model(kind)

    class SquaredScoreEmbedder(type(original)):
        def compute_score(self, q_reps, p_reps):
            assert q_reps.ndim == p_reps.ndim == 2
            return super().compute_score(q_reps, p_reps).square()

    model = SquaredScoreEmbedder(original.model, temperature=0.7)
    queries, passages = torch.randn(2, 16), torch.randn(6, 16)

    actual = model.compute_local_score(queries, passages)

    torch.testing.assert_close(actual, per_query_scores(queries, passages).square())
