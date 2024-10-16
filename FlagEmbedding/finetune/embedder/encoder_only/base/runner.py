import os
import logging
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)

from FlagEmbedding.abc.finetune.embedder import AbsRunner, AbsEmbedderModel, TrainerCallbackForDataRefresh
from FlagEmbedding.finetune.embedder.encoder_only.base.modeling import BiEncoderModel
from FlagEmbedding.finetune.embedder.encoder_only.base.trainer import EncoderOnlyTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyRunner(AbsRunner):
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=self.model_args.trust_remote_code
        )
        base_model = AutoModel.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=self.model_args.trust_remote_code
        )
        
        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=os.getenv('HF_TOKEN', None),
        )
        logger.info('Config: %s', config)
        
        model = BiEncoderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        )
        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyTrainer:
        trainer = EncoderOnlyTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(TrainerCallbackForDataRefresh(self.train_dataset))
        return trainer
