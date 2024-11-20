import os
import torch
import logging
from typing import Tuple
from transformers import (
    AutoModel, AutoConfig,
    AutoTokenizer, PreTrainedTokenizer
)
from huggingface_hub import snapshot_download

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderRunner, AbsEmbedderModel,
    AbsEmbedderDataArguments, EmbedderTrainerCallbackForDataRefresh
)
from .modeling import EncoderOnlyEmbedderM3Model
from .trainer import EncoderOnlyEmbedderM3Trainer
from .arguments import EncoderOnlyEmbedderM3ModelArguments, EncoderOnlyEmbedderM3TrainingArguments

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderM3Runner(AbsEmbedderRunner):
    """
    M3 model runner for finetuning.
    
    Args:
        model_args (EncoderOnlyEmbedderM3ModelArguments): Model arguments
        data_args (AbsEmbedderDataArguments): Data arguments.
        training_args (EncoderOnlyEmbedderM3TrainingArguments): Training arguments.
    """
    def __init__(
        self,
        model_args: EncoderOnlyEmbedderM3ModelArguments,
        data_args: AbsEmbedderDataArguments,
        training_args: EncoderOnlyEmbedderM3TrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: EncoderOnlyEmbedderM3ModelArguments
        self.data_args: AbsEmbedderDataArguments
        self.training_args: EncoderOnlyEmbedderM3TrainingArguments

    @staticmethod
    def get_model(
        model_name_or_path: str,
        trust_remote_code: bool = False,
        colbert_dim: int = -1,
        cache_dir: str = None
    ):
        """Get the model.

        Args:
            model_name_or_path (str):  If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
                load a model from HuggingFace Hub with the name.
            trust_remote_code (bool, optional): trust_remote_code to use when loading models from HF. Defaults to ``False``.
            colbert_dim (int, optional): Colbert dim to set. Defaults to ``-1``.
            cache_dir (str, optional): HF cache dir to store the model. Defaults to ``None``.

        Returns:
            dict: A dictionary containing the model, colbert linear and sparse linear.
        """
        cache_folder = os.getenv('HF_HUB_CACHE', None) if cache_dir is None else cache_dir
        if not os.path.exists(model_name_or_path):
            model_name_or_path = snapshot_download(
                repo_id=model_name_or_path,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )

        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code
        )
        colbert_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=model.config.hidden_size if colbert_dim <= 0 else colbert_dim
        )
        sparse_linear = torch.nn.Linear(
            in_features=model.config.hidden_size,
            out_features=1
        )

        colbert_model_path = os.path.join(model_name_or_path, 'colbert_linear.pt')
        sparse_model_path = os.path.join(model_name_or_path, 'sparse_linear.pt')
        if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
            logger.info('loading existing colbert_linear and sparse_linear---------')
            colbert_state_dict = torch.load(colbert_model_path, map_location='cpu', weights_only=True)
            sparse_state_dict = torch.load(sparse_model_path, map_location='cpu', weights_only=True)
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

        model = EncoderOnlyEmbedderM3Model(
            self.get_model(self.model_args.model_name_or_path, self.model_args.trust_remote_code, self.model_args.colbert_dim),
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            unified_finetuning=self.training_args.unified_finetuning,
            use_self_distill=self.training_args.use_self_distill,
            self_distill_start_step=self.training_args.self_distill_start_step
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_trainer(self) -> EncoderOnlyEmbedderM3Trainer:
        """Load the M3 trainer.

        Returns:
            EncoderOnlyEmbedderM3Trainer: M3 Trainer instance.
        """
        trainer = EncoderOnlyEmbedderM3Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        return trainer
