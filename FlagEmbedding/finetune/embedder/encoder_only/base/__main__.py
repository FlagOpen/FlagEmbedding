from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.embedder import (
    AbsModelArguments,
    AbsDataArguments,
    AbsTrainingArguments
)
from FlagEmbedding.finetune.embedder.encoder_only.base.runner import EncoderOnlyRunner


parser = HfArgumentParser((AbsModelArguments, AbsDataArguments, AbsTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: AbsModelArguments
data_args: AbsDataArguments
training_args: AbsTrainingArguments

runner = EncoderOnlyRunner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()
