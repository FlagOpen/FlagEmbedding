import logging
from typing import Union, Tuple
from FlagEmbedding.abc.evaluation import AbsEvalRunner, EvalReranker, \
    AbsEvalModelArgs as BrightEvalModelArgs

from .prompts import BrightShortInstructions, BrightLongInstructions
from .arguments import BrightEvalArgs
from .data_loader import BrightShortEvalDataLoader, BrightLongEvalDataLoader
from .searcher import BrightEvalDenseRetriever

logger = logging.getLogger(__name__)


class BrightEvalRunner(AbsEvalRunner):
    """
    Evaluation runner of Bright.
    """
    def __init__(self, eval_args: BrightEvalArgs, model_args: BrightEvalModelArgs):
        super().__init__(eval_args, model_args)
        self.eval_args: BrightEvalArgs
        self.model_args: BrightEvalModelArgs

    def load_data_loader(self) -> Union[BrightShortEvalDataLoader, BrightLongEvalDataLoader]:
        """Load the data loader instance by args.

        Returns:
            Union[BrightShortEvalDataLoader, BrightLongEvalDataLoader]: The Bright data loader instance.
        """
        if self.eval_args.task_type == "short":
            data_loader_class = BrightShortEvalDataLoader
        elif self.eval_args.task_type == "long":
            data_loader_class = BrightLongEvalDataLoader
        else:
            raise ValueError(f"Invalid task type: {self.eval_args.task_type}")

        data_loader = data_loader_class(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

    def load_retriever_and_reranker(self) -> Tuple[BrightEvalDenseRetriever, Union[EvalReranker, None]]:
        """Load retriever and reranker for evaluation

        Returns:
            Tuple[BrightEvalDenseRetriever, Union[EvalReranker, None]]: A :class:BrightEvalDenseRetriever object for retrieval, and a
                :class:EvalReranker object if reranker provided.
        """
        embedder, reranker = self.get_models(self.model_args)
        retriever = BrightEvalDenseRetriever(
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
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)

        if len(dataset_names) == 0:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the default dataset.")
            self.evaluator(
                splits=self.eval_args.splits,
                search_results_save_dir=self.eval_args.output_dir,
                retriever=self.retriever,
                reranker=self.reranker,
                corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values
            )
            logger.info(f"{self.eval_args.eval_name} evaluation completed.")
        else:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the following dataset names: {dataset_names}")
            for dataset_name in dataset_names:
                if self.eval_args.use_special_instructions:
                    self.retriever.stop_multi_process_pool()
                    if self.eval_args.task_type == "short":
                        self.retriever.embedder.query_instruction_for_retrieval = BrightShortInstructions[dataset_name]
                    elif self.eval_args.task_type == "long":
                        self.retriever.embedder.query_instruction_for_retrieval = BrightLongInstructions[dataset_name]
                    else:
                        raise ValueError(f"Invalid task type: {self.eval_args.task_type}")

                # NOTE: pass qrels to searcher to exclude documents from raw search results
                evaluator_kwargs = {}
                evaluator_kwargs["retriever_qrels"] = self.data_loader.load_qrels(dataset_name=dataset_name, split=self.eval_args.splits)

                logger.info(f"Running {self.eval_args.eval_name} evaluation on: {dataset_name}")
                self.evaluator(
                    splits=self.eval_args.splits,
                    search_results_save_dir=self.eval_args.output_dir,
                    retriever=self.retriever,
                    reranker=self.reranker,
                    corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                    ignore_identical_ids=self.eval_args.ignore_identical_ids,
                    k_values=self.eval_args.k_values,
                    dataset_name=dataset_name,
                    **evaluator_kwargs,
                )
            logger.info(f"{self.eval_args.eval_name} evaluation on {dataset_names} completed.")

        logger.info("Start computing metrics.")
        self.evaluate_metrics(
            search_results_save_dir=self.eval_args.output_dir,
            output_method=self.eval_args.eval_output_method,
            output_path=self.eval_args.eval_output_path,
            metrics=self.eval_args.eval_metrics
        )
