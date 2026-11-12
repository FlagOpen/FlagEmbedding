from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModelArguments
from runner import IREmbedderRunner
from arguments import IREmbedderTrainingArguments, IREmbedderDataArguments


if __name__ == '__main__':
    parser = HfArgumentParser((AbsEmbedderModelArguments, IREmbedderDataArguments, IREmbedderTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: AbsEmbedderModelArguments
    data_args: IREmbedderDataArguments
    training_args: IREmbedderTrainingArguments

    runner = IREmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()
