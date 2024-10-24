from dataclasses import dataclass, field
from typing import List

from FlagEmbedding.abc.evaluation.arguments import AbsEvalArgs

@dataclass
class BEIREvalArgs(AbsEvalArgs):
    dataset_names: List[str] = field(
        default=None, metadata={"help": "The dataset names to evaluate. Default: None"}
    )
    # arguana climate-fever cqadupstack dbpedia-entity fever fiqa hotpotqa msmarco nfcorpus nq quora scidocs scifact trec-covid webis-touche2020 