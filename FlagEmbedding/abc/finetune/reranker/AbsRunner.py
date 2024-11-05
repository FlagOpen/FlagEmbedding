import os
import logging
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from transformers import set_seed, PreTrainedTokenizer


from .AbsArguments import (
    AbsRerankerModelArguments,
    AbsRerankerDataArguments,
    AbsRerankerTrainingArguments
)
from .AbsTrainer import AbsRerankerTrainer
from .AbsModeling import AbsRerankerModel
from .AbsDataset import (
    AbsRerankerTrainDataset, AbsRerankerCollator,
    AbsLLMRerankerTrainDataset, AbsLLMRerankerCollator
)

logger = logging.getLogger(__name__)


class AbsRerankerRunner(ABC):
    """Abstract class to run reranker model fine-tuning.

    Args:
        model_args (AbsRerankerModelArguments): Model arguments
        data_args (AbsRerankerDataArguments): Data arguments.
        training_args (AbsRerankerTrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: AbsRerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

        # Set seed
        set_seed(training_args.seed)

        self.tokenizer, self.model = self.load_tokenizer_and_model()
        self.train_dataset = self.load_train_dataset()
        self.data_collator = self.load_data_collator()
        self.trainer = self.load_trainer()

    @abstractmethod
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Abstract method to load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsRerankerModel]: Loaded tokenizer and model instances.
        """
        pass

    @abstractmethod
    def load_trainer(self) -> AbsRerankerTrainer:
        """Abstract method to load the trainer.

        Returns:
            AbsRerankerTrainer: The loaded trainer instance.
        """
        pass

    def load_train_dataset(self) -> AbsRerankerTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
            AbsRerankerTrainDataset: The loaded dataset instance.
        """
        if self.model_args.model_type == 'encoder':
            train_dataset = AbsRerankerTrainDataset(
                args=self.data_args,
                tokenizer=self.tokenizer
            )
        else:
            train_dataset = AbsLLMRerankerTrainDataset(
                args=self.data_args,
                tokenizer=self.tokenizer
            )
        return train_dataset

    def load_data_collator(self) -> AbsRerankerCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsRerankerCollator: Loaded data collator.
        """
        if self.model_args.model_type == 'encoder':
            RerankerCollator = AbsRerankerCollator
        else:
            RerankerCollator = AbsLLMRerankerCollator

        data_collator = RerankerCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt"
        )
        return data_collator

    def run(self):
        """
        Executes the training process.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()
