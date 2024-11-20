import logging
from typing import Tuple
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder.AbsArguments import AbsEmbedderTrainingArguments
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderRunner, AbsEmbedderModel, EmbedderTrainerCallbackForDataRefresh

from .arguments import DecoderOnlyEmbedderICLModelArguments, DecoderOnlyEmbedderICLDataArguments
from .trainer import DecoderOnlyEmbedderICLTrainer
from .modeling import BiDecoderOnlyEmbedderICLModel
from .dataset import DecoderOnlyEmbedderICLSameDatasetTrainDataset
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class DecoderOnlyEmbedderICLRunner(AbsEmbedderRunner):
    """Runner class for decoder only icl model.

    Args:
        model_args (DecoderOnlyEmbedderICLModelArguments): Model arguments instance.
        data_args (DecoderOnlyEmbedderICLDataArguments): Data arguments instance.
        training_args (AbsEmbedderTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: DecoderOnlyEmbedderICLModelArguments,
        data_args: DecoderOnlyEmbedderICLDataArguments,
        training_args: AbsEmbedderTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: DecoderOnlyEmbedderICLModelArguments
        self.data_args: DecoderOnlyEmbedderICLDataArguments
        self.training_args: AbsEmbedderTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            add_eos_token=True
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.model_args.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.model_args.additional_special_tokens} already exists in the tokenizer.")
        base_model = get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer))

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        model = BiDecoderOnlyEmbedderICLModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_trainer(self) -> DecoderOnlyEmbedderICLTrainer:
        """Load the trainer.

        Returns:
            DecoderOnlyEmbedderICLTrainer: Loaded trainer instance.
        """
        trainer = DecoderOnlyEmbedderICLTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer

    def load_train_dataset(self) -> DecoderOnlyEmbedderICLSameDatasetTrainDataset:
        """Load the dataset instance for training.

        Raises:
            NotImplementedError: Only support `same_dataset_within_batch` for `DecoderOnlyEmbedderICLRunner`.

        Returns:
            DecoderOnlyEmbedderICLSameDatasetTrainDataset: The dataset instance.
        """
        if self.data_args.same_dataset_within_batch:
            train_dataset = DecoderOnlyEmbedderICLSameDatasetTrainDataset(
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
            raise NotImplementedError("Only support `same_dataset_within_batch` for `DecoderOnlyEmbedderICLRunner`.")
        return train_dataset

    def run(self):
        """
        Run the finetune.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()
        
        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
