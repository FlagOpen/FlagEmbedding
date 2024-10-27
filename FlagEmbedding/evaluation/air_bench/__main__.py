from transformers import HfArgumentParser

from .arguments import AIRBenchEvalArgs, AIRBenchEvalModelArgs
from .runner import AIRBenchEvalRunner


parser = HfArgumentParser((
    AIRBenchEvalArgs,
    AIRBenchEvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: AIRBenchEvalArgs
model_args: AIRBenchEvalModelArgs

runner = AIRBenchEvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()

print("==============================================")
print("Search results have been generated.")
print("For computing metrics, please refer to the official AIR-Bench docs:")
print("- https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/submit_to_leaderboard.md")
