import os

import torch
from transformers import AutoTokenizer

from FlagEmbedding.utils.tokenizer_compat import pad_with_compat


RERANKER_MODEL_NAME = os.environ.get(
    "FLAGEMBEDDING_RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)
ENCODER_MODEL_NAME = os.environ.get(
    "FLAGEMBEDDING_ENCODER_MODEL",
    "BAAI/bge-base-en-v1.5",
)


class MappingOnlyTokenizer:
    """Minimal tokenizer double that rejects list-of-mapping inputs."""

    def __init__(self):
        self.received = None
        self.kwargs = None

    def pad(self, encoded_inputs, **kwargs):
        if not isinstance(encoded_inputs, dict):
            raise AttributeError("'list' object has no attribute 'keys'")
        self.received = encoded_inputs
        self.kwargs = kwargs
        return encoded_inputs


def test_pad_with_compat_normalizes_list_of_mappings():
    tokenizer = MappingOnlyTokenizer()
    records = [
        {"input_ids": [101, 10, 102], "attention_mask": [1, 1, 1]},
        {"input_ids": [101, 20, 30, 102], "attention_mask": [1, 1, 1, 1]},
    ]

    result = pad_with_compat(tokenizer, records, padding=True, return_tensors="pt")

    assert result == {
        "input_ids": [[101, 10, 102], [101, 20, 30, 102]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
    }
    assert tokenizer.received == result
    assert tokenizer.kwargs == {"padding": True, "return_tensors": "pt"}


def test_pad_with_compat_passes_mapping_inputs_through():
    tokenizer = MappingOnlyTokenizer()
    encoded_inputs = {
        "input_ids": [[101, 10, 102]],
        "attention_mask": [[1, 1, 1]],
    }

    result = pad_with_compat(tokenizer, encoded_inputs, padding=True)

    assert result is encoded_inputs
    assert tokenizer.received is encoded_inputs


def test_local_bge_reranker_tokenizer_padding_compatibility():
    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME,
    )
    encoded = tokenizer(
        ["What is AI?", "Explain artificial intelligence in one sentence."],
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    records = [
        {key: encoded[key][index] for key in encoded.keys()}
        for index in range(2)
    ]

    padded = pad_with_compat(
        tokenizer,
        records,
        padding=True,
        return_tensors="pt",
    )

    assert padded["input_ids"].shape[0] == 2
    assert padded["attention_mask"].shape == padded["input_ids"].shape
    assert torch.all(padded["attention_mask"][:, 0] == 1)


def test_local_bge_encoder_token_type_ids_are_preserved():
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_MODEL_NAME,
    )
    encoded = tokenizer(
        ["short text", "a longer text for padding"],
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    records = [
        {key: encoded[key][index] for key in encoded.keys()}
        for index in range(2)
    ]

    padded = pad_with_compat(
        tokenizer,
        records,
        padding=True,
        return_tensors="pt",
    )

    assert "token_type_ids" in padded
    assert padded["token_type_ids"].shape == padded["input_ids"].shape
    assert padded["attention_mask"].shape == padded["input_ids"].shape
