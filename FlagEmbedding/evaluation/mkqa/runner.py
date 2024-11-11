from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import MKQAEvalDataLoader
from .evaluator import MKQAEvaluator


class MKQAEvalRunner(AbsEvalRunner):
    """
    Evaluation runner of MKQA.
    """
    def load_data_loader(self) -> MKQAEvalDataLoader:
        """Load the data loader instance by args.

        Returns:
            MKQAEvalDataLoader: The MKQA data loader instance.
        """
        data_loader = MKQAEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

    def load_evaluator(self) -> MKQAEvaluator:
        """Load the evaluator instance by args.

        Returns:
            MKQAEvaluator: The MKQA evaluator instance.
        """
        evaluator = MKQAEvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )
        return evaluator
