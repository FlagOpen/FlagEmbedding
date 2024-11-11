from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import MIRACLEvalDataLoader


class MIRACLEvalRunner(AbsEvalRunner):
    """
    Evaluation runner of MIRACL.
    """
    def load_data_loader(self) -> MIRACLEvalDataLoader:
        """Load the data loader instance by args.

        Returns:
            MIRACLEvalDataLoader: The MIRACL data loader instance.
        """
        data_loader = MIRACLEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader
