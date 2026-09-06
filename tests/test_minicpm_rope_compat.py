"""Exercise MiniCPM RoPE through real configuration and model loading."""

from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from FlagEmbedding.finetune.reranker.decoder_only.layerwise.arguments import (
    RerankerModelArguments,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.configuration_minicpm_reranker import (
    LayerWiseMiniCPMConfig,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.load_model import get_model
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling import (
    CrossDecoderModel,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling_minicpm_reranker import (
    LayerWiseMiniCPMForCausalLM,
)

SCALING = [None, {"type": "linear", "factor": 2.0}, {"type": "dynamic", "factor": 2.0}]


def tiny_config(scaling):
    return LayerWiseMiniCPMConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=8,
        start_layer=1,
        head_multi=True,
        use_cache=False,
        rope_theta=8192.0,
        rope_scaling=deepcopy(scaling),
        attn_implementation="eager",
    )


@pytest.mark.parametrize("scaling", SCALING)
@pytest.mark.parametrize("roundtrip", [False, True])
def test_rope_values_after_config_loading(scaling, roundtrip, tmp_path):
    config = tiny_config(scaling)
    if roundtrip:
        config.save_pretrained(tmp_path)
        config = LayerWiseMiniCPMConfig.from_pretrained(tmp_path)
    model = LayerWiseMiniCPMForCausalLM(config)
    rotary = model.model.layers[0].self_attn.rotary_emb
    # Evaluate beyond the original context length to exercise dynamic scaling.
    length, dim = 12, 8
    cos, sin = rotary(torch.zeros(1, 4, length, dim), seq_len=length)
    base = 8192.0
    positions = torch.arange(length, dtype=torch.float32)
    if scaling is not None and scaling["type"] == "linear":
        positions = positions / 2.0
    elif scaling is not None and scaling["type"] == "dynamic":
        base *= (2.0 * length / 8 - 1.0) ** (dim / (dim - 2))
    angles = positions[:, None] / base ** (
        torch.arange(0, dim, 2).float()[None, :] / dim
    )
    angles = torch.cat((angles, angles), dim=-1)
    torch.testing.assert_close(cos, angles.cos(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sin, angles.sin(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("scaling", SCALING)
def test_finetune_loader_and_training_step(scaling, tmp_path):
    torch.manual_seed(0)
    config = tiny_config(scaling)
    config.register_for_auto_class()
    model = LayerWiseMiniCPMForCausalLM(config).eval()
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    with torch.no_grad():
        model.model.norm.weight.fill_(0.75)
    model.save_pretrained(tmp_path)
    inputs = {
        "input_ids": torch.randint(0, 64, (4, 6)),
        "attention_mask": torch.ones(4, 6),
    }
    with torch.no_grad():
        expected = model(**inputs).logits
    loaded = get_model(
        RerankerModelArguments(
            model_name_or_path=str(tmp_path),
            model_type="from_finetuned_model",
            trust_remote_code=True,
            use_lora=False,
        ),
        only_for_one_logit=1,
    ).eval()
    with torch.no_grad():
        actual = loaded(**inputs).logits
    assert loaded.get_output_embeddings().weight is loaded.get_input_embeddings().weight
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    ranker = CrossDecoderModel(
        loaded, tokenizer=tokenizer, train_batch_size=2, start_layer=1
    ).train()
    optimizer = torch.optim.SGD(ranker.parameters(), lr=0.01)
    before = loaded.model.layers[0].self_attn.q_proj.weight.detach().clone()
    loss = ranker(pair=inputs).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in ranker.parameters() if p.grad is not None
    )
    optimizer.step()
    assert not torch.equal(before, loaded.model.layers[0].self_attn.q_proj.weight)


@pytest.mark.parametrize("head_type", ["raw", "simple", "complex"])
@pytest.mark.parametrize("head_multi", [False, True])
@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_checkpoint_preserves_head_weights(
    head_type, head_multi, tie_embeddings, tmp_path
):
    config = tiny_config(None)
    config.head_type = head_type
    config.head_multi = head_multi
    config.tie_word_embeddings = tie_embeddings
    config.attention_bias = True
    config.pad_token_id = 0
    model = LayerWiseMiniCPMForCausalLM(config)
    # Preserve checkpoint values even when they differ from initialization defaults.
    with torch.no_grad():
        model.model.embed_tokens.weight[0].fill_(0.25)
        model.model.layers[0].self_attn.q_proj.bias.fill_(0.5)
    model.save_pretrained(tmp_path)
    loaded = LayerWiseMiniCPMForCausalLM.from_pretrained(tmp_path)
    shared = (
        getattr(loaded.get_output_embeddings(), "weight", None)
        is loaded.get_input_embeddings().weight
    )
    assert shared == tie_embeddings
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
