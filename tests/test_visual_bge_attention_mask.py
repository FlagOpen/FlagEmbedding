"""Regression tests for Visualized-BGE multimodal mask alignment."""

import pytest
import torch

from visual_bge.modeling import build_multimodal_attention_mask


def test_multimodal_mask_matches_embedding_layout():
    prompt_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    original = prompt_mask.clone()

    result = build_multimodal_attention_mask(prompt_mask, image_token_len=2)

    expected = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]],
        dtype=torch.long,
    )
    assert torch.equal(result, expected)
    assert result.dtype == prompt_mask.dtype
    assert torch.equal(prompt_mask, original)


def test_multimodal_mask_supports_empty_image_tokens():
    prompt_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    result = build_multimodal_attention_mask(prompt_mask, image_token_len=0)

    assert torch.equal(result, prompt_mask)
    assert result.dtype == torch.bool


@pytest.mark.parametrize("bad_len", [-1, -3])
def test_multimodal_mask_rejects_negative_image_length(bad_len):
    with pytest.raises(ValueError, match="image_token_len"):
        build_multimodal_attention_mask(torch.ones(1, 2, dtype=torch.long), bad_len)

