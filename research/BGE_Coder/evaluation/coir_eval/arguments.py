from typing import List
from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation import (
    AbsEvalModelArgs as COIREvalModelArgs,
)


def coir_tasks():
    return [
        "apps",
        "codefeedback-mt",
        "codefeedback-st",
        "CodeSearchNet-ccr-go",
        "CodeSearchNet-ccr-java",
        "CodeSearchNet-ccr-javascript",
        "CodeSearchNet-ccr-php",
        "CodeSearchNet-ccr-python",
        "CodeSearchNet-ccr-ruby",
        "CodeSearchNet-go",
        "CodeSearchNet-java",
        "CodeSearchNet-javascript",
        "CodeSearchNet-php",
        "CodeSearchNet-python",
        "CodeSearchNet-ruby",
        "codetrans-contest",
        "codetrans-dl",
        "cosqa",
        "stackoverflow-qa",
        "synthetic-text2sql"
    ]


@dataclass
class COIREvalArgs:
    output_dir: str = field(
        default="./results", metadata={"help": "Path to save results."}
    )
    tasks: List[str] = field(
        default_factory=coir_tasks,
        metadata={
            "help": "Tasks to evaluate. Default: None. Available tasks: ['apps', 'codefeedback-mt', 'codefeedback-st', 'CodeSearchNet-ccr-go', 'CodeSearchNet-ccr-java', 'CodeSearchNet-ccr-javascript', 'CodeSearchNet-ccr-php', 'CodeSearchNet-ccr-python', 'CodeSearchNet-ccr-ruby', 'CodeSearchNet-go', 'CodeSearchNet-java', 'CodeSearchNet-javascript', 'CodeSearchNet-php', 'CodeSearchNet-python', 'CodeSearchNet-ruby', 'codetrans-contest', 'codetrans-dl', 'cosqa', 'stackoverflow-qa', 'synthetic-text2sql']",
            "choices": [
                "apps",
                "codefeedback-mt",
                "codefeedback-st",
                "CodeSearchNet-ccr-go",
                "CodeSearchNet-ccr-java",
                "CodeSearchNet-ccr-javascript",
                "CodeSearchNet-ccr-php",
                "CodeSearchNet-ccr-python",
                "CodeSearchNet-ccr-ruby",
                "CodeSearchNet-go",
                "CodeSearchNet-java",
                "CodeSearchNet-javascript",
                "CodeSearchNet-php",
                "CodeSearchNet-python",
                "CodeSearchNet-ruby",
                "codetrans-contest",
                "codetrans-dl",
                "cosqa",
                "stackoverflow-qa",
                "synthetic-text2sql"
            ]
        }
    )
    use_special_instructions: bool = field(
        default=False, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )