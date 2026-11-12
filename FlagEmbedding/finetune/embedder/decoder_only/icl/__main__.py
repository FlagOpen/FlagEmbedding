from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.decoder_only.icl import (
    DecoderOnlyEmbedderICLDataArguments,
    DecoderOnlyEmbedderICLTrainingArguments,
    DecoderOnlyEmbedderICLModelArguments,
    DecoderOnlyEmbedderICLRunner,
)


def main():
    parser = HfArgumentParser((
        DecoderOnlyEmbedderICLModelArguments,
        DecoderOnlyEmbedderICLDataArguments,
        DecoderOnlyEmbedderICLTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: DecoderOnlyEmbedderICLModelArguments
    data_args: DecoderOnlyEmbedderICLDataArguments
    training_args: DecoderOnlyEmbedderICLTrainingArguments

    runner = DecoderOnlyEmbedderICLRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
