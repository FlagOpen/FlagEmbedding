"""Regression tests for integer teacher scores in embedder distillation."""

from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

from FlagEmbedding.finetune.embedder.encoder_only.base.modeling import (
    BiEncoderOnlyEmbedderModel,
)
from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
    EncoderOnlyEmbedderM3Model,
)


@pytest.fixture(params=["base", "m3_dense", "m3_unified"])
def embedder(request):
    torch.manual_seed(42)
    config = XLMRobertaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    encoder = XLMRobertaModel(config)
    if request.param == "base":
        return BiEncoderOnlyEmbedderModel(encoder)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(
                {
                    "<s>": 0,
                    "<pad>": 1,
                    "</s>": 2,
                    "<unk>": 3,
                },
                unk_token="<unk>",
            )
        ),
        cls_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    return EncoderOnlyEmbedderM3Model(
        base_model={
            "model": encoder,
            "sparse_linear": torch.nn.Linear(config.hidden_size, 1),
            "colbert_linear": torch.nn.Linear(config.hidden_size, config.hidden_size),
        },
        tokenizer=tokenizer,
        unified_finetuning=request.param == "m3_unified",
    )


@pytest.mark.parametrize("kd_loss_type", ["kl_div", "m3_kd_loss"])
@pytest.mark.parametrize("teacher_scores", [[2, 0, -1, 3], [2, 0.5, -1, 3]])
def test_integer_teacher_scores_match_float_loss_and_gradients(
    embedder, kd_loss_type, teacher_scores
):
    embedder.kd_loss_type = kd_loss_type
    float_embedder = deepcopy(embedder)
    queries = {
        "input_ids": torch.tensor([[0, 4, 5, 2, 1], [0, 6, 7, 8, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
    }
    passages = {
        "input_ids": torch.tensor(
            [
                [0, 4, 9, 2, 1],
                [0, 10, 11, 2, 1],
                [0, 6, 12, 13, 2],
                [0, 14, 15, 2, 1],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ]
        ),
    }

    expected_loss = float_embedder(
        queries=queries,
        passages=passages,
        teacher_scores=[float(score) for score in teacher_scores],
    ).loss
    loss = embedder(
        queries=queries,
        passages=passages,
        teacher_scores=teacher_scores,
    ).loss
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected_loss)

    loss.backward()
    expected_loss.backward()
    for (name, param), (expected_name, expected_param) in zip(
        embedder.named_parameters(), float_embedder.named_parameters()
    ):
        assert name == expected_name
        if expected_param.grad is None:
            assert param.grad is None
        else:
            assert torch.isfinite(param.grad).all()
            torch.testing.assert_close(param.grad, expected_param.grad)
