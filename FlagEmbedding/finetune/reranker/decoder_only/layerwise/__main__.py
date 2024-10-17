from transformers import HfArgumentParser

from FlagEmbedding.abc.finetune.reranker import (
    AbsRerankerModelArguments,
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)

from FlagEmbedding.finetune.reranker.decoder_only.layerwise.runner import DecoderOnlyRerankerRunner
from FlagEmbedding.finetune.reranker.decoder_only.layerwise.arguments import RerankerModelArguments

parser = HfArgumentParser((RerankerModelArguments, AbsRerankerDataArguments, AbsRerankerTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: RerankerModelArguments
data_args: AbsRerankerDataArguments
training_args: AbsRerankerTrainingArguments

runner = DecoderOnlyRerankerRunner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()
