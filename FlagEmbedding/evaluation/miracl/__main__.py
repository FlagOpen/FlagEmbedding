from transformers import HfArgumentParser

from FlagEmbedding.evaluation.miracl import (
    MIRACLEvalArgs, MIRACLEvalModelArgs,
    MIRACLEvalRunner
)


parser = HfArgumentParser((
    MIRACLEvalArgs,
    MIRACLEvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: MIRACLEvalArgs
model_args: MIRACLEvalModelArgs

runner = MIRACLEvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
