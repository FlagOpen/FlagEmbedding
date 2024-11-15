from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.decoder_only.base import (
    DecoderOnlyEmbedderDataArguments,
    DecoderOnlyEmbedderTrainingArguments,
    DecoderOnlyEmbedderModelArguments,
    DecoderOnlyEmbedderRunner,
)


def main():
    parser = HfArgumentParser((
        DecoderOnlyEmbedderModelArguments,
        DecoderOnlyEmbedderDataArguments,
        DecoderOnlyEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: DecoderOnlyEmbedderModelArguments
    data_args: DecoderOnlyEmbedderDataArguments
    training_args: DecoderOnlyEmbedderTrainingArguments

    runner = DecoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
