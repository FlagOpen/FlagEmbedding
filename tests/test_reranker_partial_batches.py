import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from FlagEmbedding.abc.finetune.reranker.AbsDataset import (
    AbsLLMRerankerCollator,
    AbsRerankerCollator,
)
from FlagEmbedding.finetune.reranker.decoder_only.base.modeling import (
    CrossDecoderModel,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.configuration_minicpm_reranker import (
    LayerWiseMiniCPMConfig,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling import (
    CrossDecoderModel as LayerwiseReranker,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling_minicpm_reranker import (
    LayerWiseMiniCPMForCausalLM,
)
from FlagEmbedding.finetune.reranker.encoder_only.base.modeling import CrossEncoderModel


def make_model(kind):
    torch.manual_seed(19)
    vocab = ["[PAD]", "[UNK]", "Yes", "query", "positive", "negative", "other"]
    backend = Tokenizer(
        WordLevel(dict(zip(vocab, range(len(vocab)))), unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    if kind == "encoder":
        base = BertForSequenceClassification(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_labels=1,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                classifier_dropout=0.0,
                pad_token_id=0,
            )
        )
        model = CrossEncoderModel(base, tokenizer, train_batch_size=3)
        collator_type = AbsRerankerCollator
    elif kind == "decoder":
        base = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocab),
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                pad_token_id=0,
            )
        )
        model = CrossDecoderModel(base, tokenizer, train_batch_size=3)
        collator_type = AbsLLMRerankerCollator
    else:
        config = LayerWiseMiniCPMConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            start_layer=1,
            pad_token_id=0,
            use_cache=False,
            dim_model_base=16,
        )
        config._attn_implementation = "eager"
        base = LayerWiseMiniCPMForCausalLM(config)
        model = LayerwiseReranker(base, tokenizer, train_batch_size=3, start_layer=1)
        collator_type = AbsLLMRerankerCollator
    return model.train(), collator_type(tokenizer=tokenizer), tokenizer


def example(tokenizer, with_teacher):
    passages = [
        tokenizer("query " + word) for word in ["positive", "negative", "other"]
    ]
    return passages, [2.0, -1.0, 0.5] if with_teacher else None


@pytest.mark.parametrize("kind", ["encoder", "decoder", "layerwise"])
@pytest.mark.parametrize("queries", [1, 2, 3])
@pytest.mark.parametrize("with_teacher", [False, True])
def test_reranker_groups_by_the_actual_query_batch(kind, queries, with_teacher):
    model, collator, tokenizer = make_model(kind)
    # A real final DataLoader batch can contain fewer queries than configured.
    loader = DataLoader(
        [example(tokenizer, with_teacher) for _ in range(3 + queries)],
        batch_size=3,
        drop_last=False,
        collate_fn=collator,
    )
    batch = list(loader)[-1]

    result = model(**batch)

    scores = result.scores if isinstance(result.scores, list) else [result.scores]
    groups = [score.reshape(queries, 3) for score in scores]
    expected = sum(
        F.cross_entropy(group, torch.zeros(queries, dtype=torch.long))
        for group in groups
    )
    if with_teacher:
        targets = torch.softmax(torch.tensor([2.0, -1.0, 0.5]), dim=0)
        expected += sum(
            -(group.log_softmax(-1) * targets).sum(-1).mean() for group in groups
        )
    elif kind == "layerwise":
        targets = groups[-1].detach().softmax(-1)
        expected += sum(
            -(group.log_softmax(-1) * targets).sum(-1).mean() for group in groups[:-1]
        )
    torch.testing.assert_close(result.loss, expected)

    parameters = list(model.parameters())
    actual_gradients = torch.autograd.grad(
        result.loss, parameters, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    for actual, reference in zip(actual_gradients, expected_gradients):
        if reference is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("kind", ["encoder", "decoder", "layerwise"])
def test_direct_full_batch_calls_keep_the_configured_fallback(kind):
    model, collator, tokenizer = make_model(kind)
    batch = collator([example(tokenizer, False) for _ in range(3)])
    batch.pop("group_size", None)

    assert torch.isfinite(model(**batch).loss)


@pytest.mark.parametrize("kind", ["encoder", "decoder", "layerwise"])
@pytest.mark.parametrize("local_queries", [1, 3])
def test_group_size_metadata_survives_per_device_scattering(kind, local_queries):
    model, collator, tokenizer = make_model(kind)
    batch = collator([example(tokenizer, False) for _ in range(2 * local_queries)])
    # DataParallel slices tensors, but copies scalar metadata to each device.
    batch["pair"] = {
        key: value[: local_queries * 3] for key, value in batch["pair"].items()
    }

    result = model(**batch)

    scores = result.scores if isinstance(result.scores, list) else [result.scores]
    groups = [score.reshape(local_queries, 3) for score in scores]
    expected = sum(
        F.cross_entropy(group, torch.zeros(local_queries, dtype=torch.long))
        for group in groups
    )
    if kind == "layerwise":
        targets = groups[-1].detach().softmax(-1)
        expected += sum(
            -(group.log_softmax(-1) * targets).sum(-1).mean() for group in groups[:-1]
        )
    torch.testing.assert_close(result.loss, expected)
