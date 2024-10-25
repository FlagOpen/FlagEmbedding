from dataclasses import dataclass, field
from typing import List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs

@dataclass
class BEIREvalArgs(AbsEvalArgs):
    dataset_names: List[str] = field(
        default=None, metadata={"help": "The dataset names to evaluate. Default: None"}
    )
    use_special_instructions: bool = field(
        default=False, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )
    # arguana climate-fever cqadupstack dbpedia-entity fever fiqa hotpotqa msmarco nfcorpus nq quora scidocs scifact trec-covid webis-touche2020 