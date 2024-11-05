import os
import logging
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from transformers import set_seed, PreTrainedTokenizer


from .AbsArguments import (
    AbsEmbedderModelArguments,
    AbsEmbedderDataArguments,
    AbsEmbedderTrainingArguments
)
from .AbsTrainer import AbsEmbedderTrainer
from .AbsModeling import AbsEmbedderModel
from .AbsDataset import (
    AbsEmbedderTrainDataset, AbsEmbedderCollator,
    AbsEmbedderSameDatasetTrainDataset, AbsEmbedderSameDatasetCollator
)

logger = logging.getLogger(__name__)


class AbsEmbedderRunner(ABC):
    """Abstract class to run embedding model fine-tuning.

    Args:
        model_args (AbsEmbedderModelArguments): Model arguments
        data_args (AbsEmbedderDataArguments): Data arguments.
        training_args (AbsEmbedderTrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: AbsEmbedderModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: AbsEmbedderTrainingArguments
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
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Abstract method to load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Loaded tokenizer and model instances.
        """
        pass

    @abstractmethod
    def load_trainer(self) -> AbsEmbedderTrainer:
        """Abstract method to load the trainer.

        Returns:
            AbsEmbedderTrainer: The loaded trainer instance.
        """
        pass

    def load_train_dataset(self) -> AbsEmbedderTrainDataset:
        """Loads the training dataset based on data arguments.

        Returns:
            AbsEmbedderTrainDataset: The loaded dataset instance.
        """
        if self.data_args.same_dataset_within_batch:
            train_dataset = AbsEmbedderSameDatasetTrainDataset(
                args=self.data_args,
                default_batch_size=self.training_args.per_device_train_batch_size,
                seed=self.training_args.seed,
                tokenizer=self.tokenizer,
                process_index=self.training_args.process_index,
                num_processes=self.training_args.world_size
            )
            self.training_args.per_device_train_batch_size = 1
            self.training_args.dataloader_num_workers = 0   # avoid multi-processing
        else:
            train_dataset = AbsEmbedderTrainDataset(
                args=self.data_args,
                tokenizer=self.tokenizer
            )
        return train_dataset

    def load_data_collator(self) -> AbsEmbedderCollator:
        """Loads the appropriate data collator.

        Returns:
            AbsEmbedderCollator: Loaded data collator.
        """
        if self.data_args.same_dataset_within_batch:
            EmbedCollator = AbsEmbedderSameDatasetCollator
        else:
            EmbedCollator = AbsEmbedderCollator

        data_collator = EmbedCollator(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            sub_batch_size=self.training_args.sub_batch_size,
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
