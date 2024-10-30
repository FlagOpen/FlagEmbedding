from transformers import HfArgumentParser

from FlagEmbedding.evaluation.beir import (
    BEIREvalArgs, BEIREvalModelArgs,
    BEIREvalRunner
)


parser = HfArgumentParser((
    BEIREvalArgs,
    BEIREvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: BEIREvalArgs
model_args: BEIREvalModelArgs

runner = BEIREvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
