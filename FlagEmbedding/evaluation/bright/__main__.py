from transformers import HfArgumentParser

from FlagEmbedding.evaluation.bright import (
    BrightEvalArgs, BrightEvalModelArgs,
    BrightEvalRunner
)


def main():
    parser = HfArgumentParser((
        BrightEvalArgs,
        BrightEvalModelArgs
    ))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: BrightEvalArgs
    model_args: BrightEvalModelArgs

    runner = BrightEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
