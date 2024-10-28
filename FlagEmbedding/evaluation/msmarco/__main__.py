from transformers import HfArgumentParser

from FlagEmbedding.evaluation.msmarco import (
    MSMARCOEvalArgs, MSMARCOEvalModelArgs,
    MSMARCOEvalRunner
)


parser = HfArgumentParser((
    MSMARCOEvalArgs,
    MSMARCOEvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: MSMARCOEvalArgs
model_args: MSMARCOEvalModelArgs

runner = MSMARCOEvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
