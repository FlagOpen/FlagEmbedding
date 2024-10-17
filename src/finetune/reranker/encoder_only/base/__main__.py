from transformers import HfArgumentParser

from src.abc.finetune.reranker import (
    AbsRerankerModelArguments,
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)
from src.finetune.reranker.encoder_only.base.runner import EncoderOnlyRerankerRunner


parser = HfArgumentParser((AbsRerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: AbsRerankerModelArguments
data_args: AbsRerankerDataArguments
training_args: AbsRerankerTrainingArguments

runner = EncoderOnlyRerankerRunner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()
