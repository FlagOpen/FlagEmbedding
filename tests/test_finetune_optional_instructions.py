import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderDataArguments,
    AbsEmbedderSameDatasetTrainDataset,
)
from FlagEmbedding.finetune.embedder.decoder_only.icl.dataset import (
    DecoderOnlyEmbedderICLDataArguments,
    DecoderOnlyEmbedderICLSameDatasetTrainDataset,
)


@pytest.fixture
def tokenizer():
    vocabulary = [
        "[UNK]",
        "[PAD]",
        "<s>",
        "</s>",
        "<response>",
        "query",
        "positive",
        "negative",
        "body",
        "category",
        "[global]",
        "[row]",
        "[]",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["<response>"],
    )


@pytest.mark.parametrize("use_icl", [False, True])
@pytest.mark.parametrize("data_type", [None, "symmetric_sts", "symmetric_clustering"])
@pytest.mark.parametrize(
    "instruction, row_prompt, expected_prefix",
    [
        (None, None, ""),
        ("global", None, "[global] "),
        ("global", "row", "[row] "),
        ("", None, "[] "),
    ],
)
def test_same_dataset_preserves_text_without_an_instruction(
    tmp_path, tokenizer, use_icl, data_type, instruction, row_prompt, expected_prefix
):
    row = {
        "query": "query body",
        "pos": ["positive body"],
        "neg": ["negative body"],
        "batch_size": 1,
        "category": "category",
    }
    if data_type is not None:
        row["type"] = data_type
    if row_prompt is not None:
        row["prompt"] = row_prompt
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    argument_type = (
        DecoderOnlyEmbedderICLDataArguments if use_icl else AbsEmbedderDataArguments
    )
    args = argument_type(
        train_data=[str(path)],
        train_group_size=2,
        query_max_len=64,
        passage_max_len=64,
        query_instruction_for_retrieval=instruction,
        query_instruction_format="[{}] {}",
    )
    if use_icl:
        args.retrieval_use_examples = False
        args.icl_suffix_str = " <response>"
    dataset_type = (
        DecoderOnlyEmbedderICLSameDatasetTrainDataset
        if use_icl
        else AbsEmbedderSameDatasetTrainDataset
    )
    dataset = dataset_type(args=args, default_batch_size=2, seed=0, tokenizer=tokenizer)

    queries, passages, teacher_scores, no_in_batch_neg_flag = dataset[0]

    query_suffix = args.icl_suffix_str if use_icl else ""
    assert queries == [expected_prefix + "query body" + query_suffix]
    passage_prefix = expected_prefix if data_type is not None else ""
    passage_suffix = query_suffix if data_type is not None else ""
    assert passages == [
        passage_prefix + "positive body" + passage_suffix,
        passage_prefix + "negative body" + passage_suffix,
    ]
    assert teacher_scores is None
    assert no_in_batch_neg_flag is False
