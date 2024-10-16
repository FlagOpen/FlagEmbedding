from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.embedder import (
    AbsDataArguments,
    AbsTrainingArguments
)
from FlagEmbedding.finetune.embedder.decoder_only.base.arguments import ModelArguments
from FlagEmbedding.finetune.embedder.decoder_only.base.runner import DecoderOnlyRunner


parser = HfArgumentParser((ModelArguments, AbsDataArguments, AbsTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: ModelArguments
data_args: AbsDataArguments
training_args: AbsTrainingArguments

runner = DecoderOnlyRunner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()
