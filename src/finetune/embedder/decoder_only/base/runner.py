import logging
from typing import Tuple
from pathlib import Path
from src.abc.finetune.embedder.AbsArguments import AbsDataArguments, AbsModelArguments, AbsTrainingArguments
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from src.abc.finetune.embedder import AbsRunner, AbsEmbedderModel
from src.finetune.embedder.decoder_only.base.modeling import BiEncoderModel
from src.finetune.embedder.decoder_only.base.arguments import ModelArguments
from src.finetune.embedder.decoder_only.base.trainer import DecoderOnlyTrainer
from src.finetune.embedder.decoder_only.base.load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class DecoderOnlyRunner(AbsRunner):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: AbsDataArguments,
        training_args: AbsTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
    
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
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
        base_model = get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer))
        
        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
        )
        logger.info('Config: %s', config)
        
        model = BiEncoderModel(
            base_model=base_model,
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

    def load_trainer(self) -> DecoderOnlyTrainer:
        trainer = DecoderOnlyTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        return trainer

    def run(self):
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()
        
        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
