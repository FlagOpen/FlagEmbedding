from typing import Union, Tuple
from air_benchmark import AIRBench

from FlagEmbedding.abc.evaluation import (
    AbsEvalRunner,
    EvalDenseRetriever, EvalReranker
)

from .arguments import AIRBenchEvalArgs, AIRBenchEvalModelArgs


class AIRBenchEvalRunner:
    """
    Evaluation runner for AIR Bench.
    
    Args:
        eval_args (AIRBenchEvalArgs): :class:AIRBenchEvalArgs object with the evaluation arguments.
        model_args (AIRBenchEvalModelArgs): :class:AIRBenchEvalModelArgs object with the model arguments.
    """
    def __init__(
        self,
        eval_args: AIRBenchEvalArgs,
        model_args: AIRBenchEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args
        self.model_args.cache_dir = model_args.model_cache_dir

        self.retriever, self.reranker = self.load_retriever_and_reranker()

    def load_retriever_and_reranker(self) -> Tuple[EvalDenseRetriever, Union[EvalReranker, None]]:
        """Load retriever and reranker for evaluation

        Returns:
            Tuple[EvalDenseRetriever, Union[EvalReranker, None]]: A :class:EvalDenseRetriever object for retrieval, and a
                :class:EvalReranker object if reranker provided.
        """
        embedder, reranker = AbsEvalRunner.get_models(self.model_args)
        retriever = EvalDenseRetriever(
            embedder,
            search_top_k=self.eval_args.search_top_k,
            overwrite=self.eval_args.overwrite
        )
        if reranker is not None:
            reranker = EvalReranker(reranker, rerank_top_k=self.eval_args.rerank_top_k)
        return retriever, reranker

    def run(self):
        """
        Run the whole evaluation.
        """
        evaluation = AIRBench(
            benchmark_version=self.eval_args.benchmark_version,
            task_types=self.eval_args.task_types,
            domains=self.eval_args.domains,
            languages=self.eval_args.languages,
            splits=self.eval_args.splits,
            cache_dir=self.eval_args.cache_dir,
        )
        evaluation.run(
            self.retriever,
            reranker=self.reranker,
            output_dir=self.eval_args.output_dir,
            overwrite=self.eval_args.overwrite,
        )
