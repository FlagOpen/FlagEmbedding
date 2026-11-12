from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
    EncoderOnlyEmbedderM3DataArguments,
    EncoderOnlyEmbedderM3TrainingArguments,
    EncoderOnlyEmbedderM3ModelArguments,
    EncoderOnlyEmbedderM3Runner,
)


def main():
    parser = HfArgumentParser((EncoderOnlyEmbedderM3ModelArguments, EncoderOnlyEmbedderM3DataArguments, EncoderOnlyEmbedderM3TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: EncoderOnlyEmbedderM3ModelArguments
    data_args: EncoderOnlyEmbedderM3DataArguments
    training_args: EncoderOnlyEmbedderM3TrainingArguments

    runner = EncoderOnlyEmbedderM3Runner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
