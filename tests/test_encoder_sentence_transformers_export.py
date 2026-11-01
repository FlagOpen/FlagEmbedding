"""Check exported dense embeddings against the fine-tuned encoder."""

import pytest
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
    XLMRobertaConfig,
    XLMRobertaModel,
)

from FlagEmbedding.finetune.embedder.encoder_only.base.modeling import (
    BiEncoderOnlyEmbedderModel,
)
from FlagEmbedding.finetune.embedder.encoder_only.base.trainer import (
    EncoderOnlyEmbedderTrainer,
)
from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import (
    EncoderOnlyEmbedderM3Model,
)
from FlagEmbedding.finetune.embedder.encoder_only.m3.trainer import (
    EncoderOnlyEmbedderM3Trainer,
)


@pytest.fixture
def tokenizer():
    vocabulary = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
        )
    }
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        eos_token="[SEP]",
        model_max_length=32,
    )


def make_model(tokenizer, kind, pooling, normalize):
    torch.manual_seed(42)
    model_type = BertModel if kind == "base" else XLMRobertaModel
    config_type = BertConfig if kind == "base" else XLMRobertaConfig
    transformer = model_type(
        config_type(
            vocab_size=len(tokenizer),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            pad_token_id=tokenizer.pad_token_id,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    )
    kwargs = {
        "tokenizer": tokenizer,
        "sentence_pooling_method": pooling,
        "normalize_embeddings": normalize,
    }
    if kind == "base":
        return BiEncoderOnlyEmbedderModel(transformer, **kwargs)
    return EncoderOnlyEmbedderM3Model(
        {
            "model": transformer,
            "colbert_linear": torch.nn.Linear(8, 8),
            "sparse_linear": torch.nn.Linear(8, 1),
        },
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["base", "m3"])
@pytest.mark.parametrize("pooling", ["cls", "mean", "last_token"])
@pytest.mark.parametrize("normalize", [False, True])
def test_finetuned_encoder_round_trips_through_sentence_transformers(
    tmp_path, tokenizer, kind, pooling, normalize
):
    model = make_model(tokenizer, kind, pooling, normalize)
    texts = ["hello world", "world", "hello hello world"]
    features = tokenizer(texts, padding=True, return_tensors="pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    original_weights = model.model.get_input_embeddings().weight.detach().clone()
    loss = model(queries=features, passages=features).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(original_weights, model.model.get_input_embeddings().weight)
    model.eval()
    with torch.no_grad():
        expected = model.encode(features)
        if kind == "m3":
            expected = expected[0]

    trainer_type = (
        EncoderOnlyEmbedderTrainer if kind == "base" else EncoderOnlyEmbedderM3Trainer
    )
    trainer = trainer_type(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        processing_class=tokenizer,
    )
    trainer.save_model(str(tmp_path))
    restored = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)
    actual = restored.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if kind == "m3":
        for name in ("colbert_linear", "sparse_linear"):
            saved = torch.load(tmp_path / f"{name}.pt", weights_only=True)
            for key, value in getattr(model, name).state_dict().items():
                torch.testing.assert_close(saved[key], value)


@pytest.mark.parametrize("kind", ["base", "m3"])
def test_saving_again_updates_pooling_and_normalization(tmp_path, tokenizer, kind):
    model = make_model(tokenizer, kind, "cls", True)
    tokenizer.save_pretrained(tmp_path)
    model.save(str(tmp_path))
    model.sentence_pooling_method = "mean"
    model.normalize_embeddings = False
    model.save(str(tmp_path))
    model.eval()
    texts = ["hello world", "hello"]
    with torch.no_grad():
        expected = model.encode(tokenizer(texts, padding=True, return_tensors="pt"))
        if kind == "m3":
            expected = expected[0]
    restored = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)
    actual = restored.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
