import logging
from typing import Tuple
from pathlib import Path
from FlagEmbedding.abc.finetune.reranker.AbsArguments import AbsRerankerDataArguments, AbsRerankerTrainingArguments
from transformers import (
    AutoTokenizer, PreTrainedTokenizer
)

from FlagEmbedding.abc.finetune.reranker import AbsRerankerRunner, AbsRerankerModel

from .modeling import CrossDecoderModel
from .arguments import RerankerModelArguments
from .trainer import DecoderOnlyRerankerTrainer
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class DecoderOnlyRerankerRunner(AbsRerankerRunner):
    """
    Decoder only reranker runner for finetuning.
    
    Args:
        model_args (RerankerModelArguments): Model arguments instance.
        data_args (AbsRerankerDataArguments): Data arguments instance.
        training_args (AbsRerankerTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: RerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            add_eos_token=False,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token = tokenizer.eod
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token = tokenizer.im_start
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token = tokenizer.im_end
                tokenizer.eos_token_id = tokenizer.im_end_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        # if 'mistral' in self.model_args.model_name_or_path.lower():
        tokenizer.padding_side = 'left'

        base_model = get_model(self.model_args)

        model = CrossDecoderModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        return tokenizer, model

    def load_trainer(self) -> DecoderOnlyRerankerTrainer:
        """Load the trainer.

        Returns:
            DecoderOnlyRerankerTrainer: Loaded trainer instance.
        """
        trainer = DecoderOnlyRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        return trainer

    def run(self):
        """
        Run the finetuning.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()

        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
