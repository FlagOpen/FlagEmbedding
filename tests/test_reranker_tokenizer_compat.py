import os
from types import SimpleNamespace

from transformers import AutoTokenizer

from FlagEmbedding.abc.finetune.reranker.AbsDataset import AbsRerankerTrainDataset
from FlagEmbedding.utils.tokenizer_compat import prepare_for_model_compat


RERANKER_MODEL_NAME = os.environ.get(
    "FLAGEMBEDDING_RERANKER_MODEL",
    "BAAI/bge-reranker-base",
)


class LegacyTokenizer:
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    truncation_side = "right"

    def __init__(self):
        self.calls = []

    def prepare_for_model(self, first_ids, second_ids, **kwargs):
        self.calls.append((first_ids, second_ids, kwargs))
        return {"input_ids": list(first_ids) + list(second_ids)}


def test_encoder_pair_preserves_v5_tokenizer_input_ids():
    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME,
    )

    query = "😀🧪"
    passage = "unknown token"
    query_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    passage_ids = tokenizer(passage, add_special_tokens=False)["input_ids"]

    actual = prepare_for_model_compat(
        tokenizer,
        query_ids,
        passage_ids,
        truncation="only_second",
        max_length=64,
        padding=False,
    )
    expected = tokenizer(
        query,
        passage,
        truncation="only_second",
        max_length=64,
        padding=False,
    )

    assert not hasattr(tokenizer, "prepare_for_model")
    assert actual["input_ids"] == expected["input_ids"]
    assert actual["attention_mask"] == expected["attention_mask"]


def test_encoder_training_example_works_without_prepare_for_model():
    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME,
    )
    dataset = AbsRerankerTrainDataset.__new__(AbsRerankerTrainDataset)
    dataset.tokenizer = tokenizer
    dataset.args = SimpleNamespace(query_max_len=32, passage_max_len=128)

    actual = dataset.create_one_example("What is AI?", "AI is artificial intelligence.")
    expected = tokenizer(
        "What is AI?",
        "AI is artificial intelligence.",
        truncation="only_second",
        max_length=160,
        padding=False,
    )

    assert actual["input_ids"] == expected["input_ids"]
    assert actual["attention_mask"] == expected["attention_mask"]


def test_legacy_tokenizer_path_is_preserved():
    tokenizer = LegacyTokenizer()
    actual = prepare_for_model_compat(
        tokenizer,
        [1, 2],
        [3, 4],
        truncation="only_second",
        max_length=8,
        padding=False,
    )

    assert actual["input_ids"] == [1, 2, 3, 4]
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0][2]["add_special_tokens"] is True


def test_decoder_only_truncation_respects_tokenizer_side():
    tokenizer = LegacyTokenizer()
    tokenizer.prepare_for_model = None

    right = prepare_for_model_compat(
        tokenizer,
        [1, 2],
        [3, 4, 5],
        truncation="only_second",
        max_length=4,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,
    )
    assert right["input_ids"] == [1, 2, 3, 4]

    tokenizer.truncation_side = "left"
    left = prepare_for_model_compat(
        tokenizer,
        [1, 2],
        [3, 4, 5],
        truncation="only_second",
        max_length=4,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,
    )
    assert left["input_ids"] == [1, 2, 4, 5]
