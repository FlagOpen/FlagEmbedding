from transformers import HfArgumentParser

from FlagEmbedding.evaluation.mkqa import (
    MKQAEvalArgs, MKQAEvalModelArgs,
    MKQAEvalRunner
)


def main():
    parser = HfArgumentParser((
        MKQAEvalArgs,
        MKQAEvalModelArgs
    ))

    eval_args, model_args = parser.parse_args_into_dataclasses()
    eval_args: MKQAEvalArgs
    model_args: MKQAEvalModelArgs

    runner = MKQAEvalRunner(
        eval_args=eval_args,
        model_args=model_args
    )

    runner.run()


if __name__ == "__main__":
    main()
