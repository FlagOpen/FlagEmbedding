import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from FlagEmbedding.finetune.reranker.decoder_only.layerwise.configuration_minicpm_reranker import (
    LayerWiseMiniCPMConfig,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling import (
    CrossDecoderModel,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling_minicpm_reranker import (
    LayerWiseMiniCPMForCausalLM,
)


def make_reranker(start_layer, head_multi):
    torch.manual_seed(21)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1, "Yes": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    config = LayerWiseMiniCPMConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        start_layer=1,
        head_multi=head_multi,
        head_type="simple",
        pad_token_id=0,
        use_cache=False,
        dim_model_base=16,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    backbone = LayerWiseMiniCPMForCausalLM(config)
    model = CrossDecoderModel(
        backbone, tokenizer, train_batch_size=2, start_layer=start_layer
    )
    pairs = {
        "input_ids": torch.tensor(
            [[2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 4, 5], [2, 4, 6], [2, 4, 7]]
        ),
        "attention_mask": torch.ones(6, 3, dtype=torch.long),
    }
    return model, pairs


@pytest.mark.parametrize("head_multi", [False, True])
@pytest.mark.parametrize("start_layer", [1, 2, 3])
@pytest.mark.parametrize("with_teacher", [False, True])
def test_training_optimizes_every_requested_layer(
    start_layer, head_multi, with_teacher
):
    model, pairs = make_reranker(start_layer, head_multi)
    model.train()
    teacher_scores = [2.0, -1.0, 0.5, 1.0, 0.0, -0.5] if with_teacher else None

    actual = model(pair=pairs, teacher_scores=teacher_scores)

    requested_layers = list(range(start_layer, 4))
    reference = model.model(
        **pairs, cutoff_layers=requested_layers, output_hidden_states=True
    )
    groups = [logits[:, -1].reshape(2, 3) for logits in reference.logits]
    expected = sum(
        F.cross_entropy(group, torch.zeros(2, dtype=torch.long)) for group in groups
    )
    if with_teacher:
        targets = torch.tensor(teacher_scores).reshape(2, 3).softmax(-1)
        students = groups
    else:
        targets = groups[-1].detach().softmax(-1)
        students = groups[:-1]
    expected += sum(
        -(student.log_softmax(-1) * targets).sum(-1).mean() for student in students
    )

    assert len(actual.scores) == len(requested_layers)
    for scores, group in zip(actual.scores, groups):
        torch.testing.assert_close(scores.reshape(2, 3), group)
    torch.testing.assert_close(actual.loss, expected)
    actual.loss.backward()
    parameters = list(model.parameters())
    reference_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    for parameter, gradient in zip(parameters, reference_gradients):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient)
    if head_multi:
        for layer in requested_layers:
            gradient = model.model.lm_head[layer - 1].linear_head.weight.grad
            assert gradient is not None
            assert gradient.norm() > 0


@pytest.mark.parametrize("head_multi", [False, True])
@pytest.mark.parametrize("start_layer", [1, 2, 3])
def test_evaluation_returns_the_requested_layers_without_a_loss(
    start_layer, head_multi
):
    model, pairs = make_reranker(start_layer, head_multi)
    model.eval()

    with torch.no_grad():
        actual = model(pair=pairs)
        reference = model.model(**pairs, cutoff_layers=list(range(start_layer, 4)))

    assert actual.loss is None
    assert len(actual.scores) == 4 - start_layer
    for scores, logits in zip(actual.scores, reference.logits):
        torch.testing.assert_close(scores, logits[:, -1])


@pytest.mark.parametrize("head_multi", [False, True])
def test_backbone_default_still_returns_only_the_final_layer(head_multi):
    model, pairs = make_reranker(1, head_multi)

    assert len(model.model(**pairs).logits) == 1
