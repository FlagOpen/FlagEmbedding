from transformers import HfArgumentParser

from FlagEmbedding.evaluation.custom import (
    CustomEvalArgs, CustomEvalModelArgs,
    CustomEvalRunner
)


parser = HfArgumentParser((
    CustomEvalArgs,
    CustomEvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: CustomEvalArgs
model_args: CustomEvalModelArgs

runner = CustomEvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
