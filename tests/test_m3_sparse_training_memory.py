"""Native M3 sparse pooling: definition, gradients, and allocation budget."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BertConfig, BertModel, BertTokenizerFast

from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import EncoderOnlyEmbedderM3Model


def tiny_model(dtype=torch.float32, vocab_size=128):
    torch.manual_seed(7)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({f"word{i}": i for i in range(4, vocab_size)})
    tokenizer = BertTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", eos_token="[SEP]",
    )
    config = BertConfig(
        vocab_size=vocab_size, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2,
        hidden_dropout_prob=0., attention_probs_dropout_prob=0., pad_token_id=0,
    )
    model = EncoderOnlyEmbedderM3Model(
        {"model": BertModel(config), "sparse_linear": torch.nn.Linear(8, 1),
         "colbert_linear": torch.nn.Linear(8, 2)},
        tokenizer=tokenizer,
    ).to(dtype=dtype)
    with torch.no_grad():
        model.sparse_linear.weight.zero_()
        model.sparse_linear.weight[0, 0] = 1.
        model.sparse_linear.bias.zero_()
    return model


def token_dictionary_oracle(model, hidden, ids):
    """Visit occurrences of each vocabulary word; first occurrence wins ties."""
    weights = torch.relu(model.sparse_linear(hidden)).squeeze(-1)
    excluded = {model.tokenizer.cls_token_id, model.tokenizer.eos_token_id,
                model.tokenizer.pad_token_id, model.tokenizer.unk_token_id}
    rows = []
    for row, tokens in enumerate(ids.tolist()):
        entries = []
        for word in range(model.vocab_size):
            positions = [index for index, token in enumerate(tokens) if token == word]
            if not positions or word in excluded:
                entries.append(weights.new_zeros(()))
            else:
                entries.append(weights[row, positions].max(dim=0).values)
        rows.append(torch.stack(entries))
    return torch.stack(rows)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pattern", ["distinct", "positive_ties", "negative_and_zero"])
def test_sparse_training_values_and_first_maximum_gradients(dtype, pattern):
    model = tiny_model(dtype)
    ids = torch.tensor([[2, 5, 6, 5, 7, 6, 3, 0], [2, 7, 7, 8, 9, 8, 3, 1]])
    hidden = torch.arange(128, dtype=dtype).reshape(2, 8, 8) / 64.
    if pattern == "distinct":
        hidden[:, :, 0] = torch.tensor([[1., 2., 3., 4., 1., 2., 5., 6.],
                                       [1., 3., 2., 4., 1., 2., 5., 6.]])
    elif pattern == "positive_ties":
        hidden[:, :, 0] = 2.
    else:
        hidden[:, :, 0] = torch.tensor([[1., -2., 0., -1., 3., 0., 5., 6.],
                                       [1., 0., -2., 1., 3., -1., 5., 6.]])
    hidden.requires_grad_()
    actual = model._sparse_embedding(hidden, ids)
    expected = token_dictionary_oracle(model, hidden, ids)
    torch.testing.assert_close(actual, expected)
    coefficients = torch.linspace(.1, 1.3, model.vocab_size, dtype=dtype)[None, :]
    parameters = (hidden, model.sparse_linear.weight, model.sparse_linear.bias)
    actual_gradients = torch.autograd.grad((actual * coefficients).sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad((expected * coefficients).sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.count_nonzero(actual[:, [0, 1, 2, 3]]) == 0
    assert torch.count_nonzero(actual[:, 10:]) == 0
    if pattern == "positive_ties":
        assert actual_gradients[0][0, 1, 0] > 0
        assert actual_gradients[0][0, 3, 0] == 0


def test_sparse_training_retains_nan_propagation():
    model = tiny_model()
    ids = torch.tensor([[2, 5, 5, 6, 3]])
    hidden = torch.ones(1, 5, 8)
    hidden[0, 2, 0] = float("nan")
    actual = model._sparse_embedding(hidden, ids)
    expected = token_dictionary_oracle(model, hidden, ids)
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("sub_batch_size", [-1, 1])
def test_native_encoder_sparse_backward_matches_dictionary_definition(sub_batch_size):
    model = tiny_model()
    model.sub_batch_size = sub_batch_size
    ids = torch.tensor([[2, 5, 6, 5, 3, 0], [2, 7, 7, 8, 3, 0]])
    features = {"input_ids": ids, "attention_mask": ids.ne(0).long()}
    actual = model.encode(features)[1]
    hidden = model.model(**features, return_dict=True).last_hidden_state
    expected = token_dictionary_oracle(model, hidden, ids)
    torch.testing.assert_close(actual, expected)
    coefficients = torch.linspace(.1, 1.3, model.vocab_size)[None, :]
    parameters = (model.model.embeddings.word_embeddings.weight,
                  model.sparse_linear.weight, model.sparse_linear.bias)
    actual_gradients = torch.autograd.grad((actual * coefficients).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected * coefficients).sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("sequence_length", [64, 96])
def test_sparse_training_allocation_scales_with_tokens_plus_vocabulary(sequence_length):
    model = tiny_model()
    batch_size = 2
    ids = torch.arange(sequence_length).remainder(model.vocab_size).repeat(batch_size, 1)
    hidden = torch.ones(batch_size, sequence_length, 8, requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as profile:
        sparse = model._sparse_embedding(hidden, ids)
        sparse.sum().backward()
    # A generous constant-size multiple of the input token count and output vocabulary.
    # The original batch x sequence x vocabulary allocation exceeds this budget.
    budget = 16 * batch_size * (sequence_length + model.vocab_size)
    largest_allocation = max(event.cpu_memory_usage for event in profile.events())
    assert largest_allocation <= budget, (largest_allocation, budget)
    assert torch.isfinite(hidden.grad).all()
