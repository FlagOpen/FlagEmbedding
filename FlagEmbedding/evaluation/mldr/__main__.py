from transformers import HfArgumentParser

from FlagEmbedding.evaluation.mldr import (
    MLDREvalArgs, MLDREvalModelArgs,
    MLDREvalRunner
)


parser = HfArgumentParser((
    MLDREvalArgs,
    MLDREvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: MLDREvalArgs
model_args: MLDREvalModelArgs

runner = MLDREvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
