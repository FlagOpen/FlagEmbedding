from transformers import HfArgumentParser

from src.abc.finetune.embedder import AbsDataArguments
from src.finetune.embedder.encoder_only.m3.runner import EncoderOnlyM3Runner
from src.finetune.embedder.encoder_only.m3.arguments import M3ModelArguments, M3TrainingArguments


parser = HfArgumentParser((M3ModelArguments, AbsDataArguments, M3TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: M3ModelArguments
data_args: AbsDataArguments
training_args: M3TrainingArguments

runner = EncoderOnlyM3Runner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()
