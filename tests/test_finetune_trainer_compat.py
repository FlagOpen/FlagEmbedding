"""Regression tests for the Transformers Trainer API migration."""

from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from FlagEmbedding.finetune.embedder.decoder_only.base.trainer import (
    DecoderOnlyEmbedderTrainer,
)
from FlagEmbedding.finetune.embedder.decoder_only.icl.trainer import (
    DecoderOnlyEmbedderICLTrainer,
)
from FlagEmbedding.finetune.embedder.encoder_only.base.trainer import (
    EncoderOnlyEmbedderTrainer,
)
from FlagEmbedding.finetune.embedder.encoder_only.m3.trainer import (
    EncoderOnlyEmbedderM3Trainer,
)
from FlagEmbedding.finetune.reranker.decoder_only.base.trainer import (
    DecoderOnlyRerankerTrainer,
)
from FlagEmbedding.finetune.reranker.decoder_only.base.runner import (
    DecoderOnlyRerankerRunner,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.trainer import (
    DecoderOnlyRerankerTrainer as DecoderOnlyRerankerLayerwiseTrainer,
)
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.runner import (
    DecoderOnlyRerankerRunner as DecoderOnlyRerankerLayerwiseRunner,
)
from FlagEmbedding.finetune.reranker.encoder_only.base.trainer import (
    EncoderOnlyRerankerTrainer,
)
from FlagEmbedding.finetune.reranker.encoder_only.base.runner import (
    EncoderOnlyRerankerRunner,
)


class RecordingProcessor:
    def __init__(self, marker: str):
        self.marker = marker

    def save_pretrained(self, output_dir):
        Path(output_dir, "processor.marker").write_text(self.marker, encoding="utf-8")


class SaveableModel(torch.nn.Module):
    def save(self, output_dir):
        Path(output_dir, "model.marker").write_text("save", encoding="utf-8")

    def save_pretrained(self, output_dir):
        Path(output_dir, "model.marker").write_text("save_pretrained", encoding="utf-8")


def training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path / "output"),
        report_to=[],
    )


def test_processing_class_and_legacy_tokenizer_are_accepted(tmp_path):
    processor = RecordingProcessor("processing")
    trainer = EncoderOnlyEmbedderTrainer(
        model=SaveableModel(),
        args=training_args(tmp_path),
        processing_class=processor,
    )
    assert trainer.processing_class is processor

    legacy_processor = RecordingProcessor("legacy")
    legacy_trainer = EncoderOnlyEmbedderTrainer(
        model=SaveableModel(),
        args=training_args(tmp_path / "legacy"),
        tokenizer=legacy_processor,
    )
    assert legacy_trainer.processing_class is legacy_processor


def test_all_flagembedding_finetune_trainers_save_the_processing_class(tmp_path):
    trainer_types = [
        DecoderOnlyEmbedderTrainer,
        DecoderOnlyEmbedderICLTrainer,
        EncoderOnlyEmbedderTrainer,
        EncoderOnlyEmbedderM3Trainer,
        DecoderOnlyRerankerTrainer,
        DecoderOnlyRerankerLayerwiseTrainer,
        EncoderOnlyRerankerTrainer,
    ]

    for trainer_type in trainer_types:
        output_dir = tmp_path / trainer_type.__name__
        processor = RecordingProcessor(trainer_type.__name__)
        trainer = trainer_type(
            model=SaveableModel(),
            args=training_args(output_dir),
            processing_class=processor,
        )

        trainer._save(str(output_dir))

        assert Path(output_dir, "processor.marker").read_text(encoding="utf-8") == trainer_type.__name__
        assert Path(output_dir, "training_args.bin").exists()


def test_reranker_runners_pass_processing_class(tmp_path):
    runner_types = [
        EncoderOnlyRerankerRunner,
        DecoderOnlyRerankerRunner,
        DecoderOnlyRerankerLayerwiseRunner,
    ]

    for runner_type in runner_types:
        runner = runner_type.__new__(runner_type)
        processor = RecordingProcessor(runner_type.__name__)
        runner.model = SaveableModel()
        runner.training_args = training_args(tmp_path / runner_type.__name__)
        runner.train_dataset = None
        runner.data_collator = None
        runner.tokenizer = processor

        trainer = runner.load_trainer()

        assert trainer.processing_class is processor


def test_legacy_transformers_trainer_constructor_path(monkeypatch, tmp_path):
    """The compatibility shim uses tokenizer= with pre-v5 Trainer APIs."""
    captured = {}

    def legacy_init(self, *args, tokenizer=None, **kwargs):
        captured["tokenizer"] = tokenizer
        self.tokenizer = tokenizer
        self.model = kwargs.get("model")
        self.args = kwargs.get("args")

    monkeypatch.setattr(Trainer, "__init__", legacy_init)

    from FlagEmbedding.abc.finetune.embedder.AbsTrainer import AbsEmbedderTrainer

    class LegacyTrainer(AbsEmbedderTrainer):
        def _save(self, output_dir=None, state_dict=None):
            pass

    processor = RecordingProcessor("legacy-transformers")
    LegacyTrainer(
        model=SaveableModel(),
        args=training_args(tmp_path),
        processing_class=processor,
    )

    assert captured["tokenizer"] is processor
