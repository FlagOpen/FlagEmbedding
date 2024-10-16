import os
import torch
import logging
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)

from FlagEmbedding.abc.finetune.embedder import (
    AbsRunner, AbsEmbedderModel,
    AbsDataArguments, TrainerCallbackForDataRefresh
)
from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import M3Model
from FlagEmbedding.finetune.embedder.encoder_only.m3.trainer import EncoderOnlyM3Trainer
from FlagEmbedding.finetune.embedder.encoder_only.m3.arguments import M3ModelArguments, M3TrainingArguments

logger = logging.getLogger(__name__)


class EncoderOnlyM3Runner(AbsRunner):
    def __init__(
        self,
        model_args: M3ModelArguments,
        data_args: AbsDataArguments,
        training_args: M3TrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
    
    @staticmethod
    def get_model(
        model_name_or_path: str,
        trust_remote_code: bool = False,
        colbert_dim: int = -1
    ):
        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        colbert_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=model.config.hidden_size if colbert_dim > 0 else colbert_dim
        )
        sparse_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=1
        )
        
        colbert_model_path = os.path.join(model_name_or_path, 'colbert_linear.pt')
        sparse_model_path = os.path.join(model_name_or_path, 'sparse_linear.pt')
        if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
            logger.info('loading existing colbert_linear and sparse_linear---------')
            colbert_state_dict = torch.load(colbert_model_path, map_location='cpu')
            sparse_state_dict = torch.load(sparse_model_path, map_location='cpu')
            colbert_linear.load_state_dict(colbert_state_dict)
            sparse_linear.load_state_dict(sparse_state_dict)
        else:
            logger.info('The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')
        
        return {
            'model': model,
            'colbert_linear': colbert_linear,
            'sparse_linear': sparse_linear
        }
    
    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        tokenizer = AutoTokenizer.from_pretrained(
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
        
        model = M3Model(
            self.get_model(self.model_args.model_name_or_path, self.model_args.trust_remote_code, self.model_args.colbert_dim),
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            unified_finetuning=self.training_args.unified_finetuning,
            use_self_distill=self.training_args.use_self_distill,
            self_distill_start_step=self.training_args.self_distill_start_step
        )
        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyM3Trainer:
        trainer = EncoderOnlyM3Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(TrainerCallbackForDataRefresh(self.train_dataset))
        return trainer
