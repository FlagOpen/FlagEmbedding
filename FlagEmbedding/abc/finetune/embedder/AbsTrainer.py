import logging
from typing import Optional
from abc import ABC, abstractmethod
from transformers.trainer import Trainer
from sentence_transformers import SentenceTransformer, models
# from transformers.trainer import *

logger = logging.getLogger(__name__)


class AbsEmbedderTrainer(ABC, Trainer):
    """
    Abstract class for the trainer of embedder.
    """
    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        
        Args:
            model (AbsEmbedderModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss.
        
        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, EmbedderOutput)]: The computed loss. If ``return_outputs`` is ``True``, 
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normalized: bool = True):
        word_embedding_model = models.Transformer(ckpt_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        if normalized:
            normalize_layer = models.Normalize()
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_layer], device='cpu')
        else:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
        model.save(ckpt_dir)
