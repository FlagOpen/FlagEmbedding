import logging
from typing import Tuple
from transformers import (
    AutoModelForSequenceClassification, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)

from FlagEmbedding.abc.finetune.reranker import AbsRerankerRunner, AbsRerankerModel
from FlagEmbedding.finetune.reranker.encoder_only.base.modeling import CrossEncoderModel
from FlagEmbedding.finetune.reranker.encoder_only.base.trainer import EncoderOnlyRerankerTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyRerankerRunner(AbsRerankerRunner):
    """
    Encoder only reranker runner for finetuning.
    """
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code
        )

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            trust_remote_code=self.model_args.trust_remote_code
        )

        model = CrossEncoderModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyRerankerTrainer:
        """Load the trainer.

        Returns:
            EncoderOnlyRerankerTrainer: Loaded trainer instance.
        """
        trainer = EncoderOnlyRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        return trainer
