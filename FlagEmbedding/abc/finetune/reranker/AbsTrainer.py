import logging
import inspect
from typing import Optional
from abc import ABC, abstractmethod
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


class AbsRerankerTrainer(ABC, Trainer):
    """
    Abstract class for the trainer of reranker.
    """
    def __init__(
        self,
        *args,
        processing_class=None,
        tokenizer=None,
        **kwargs,
    ):
        """Initialize ``Trainer`` across Transformers API versions."""
        if processing_class is None:
            processing_class = tokenizer

        trainer_parameters = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_parameters:
            kwargs["processing_class"] = processing_class
        else:
            kwargs["tokenizer"] = processing_class

        super().__init__(*args, **kwargs)

    def _save_processing_class(self, output_dir: str):
        """Save the tokenizer/processor using the active Transformers API."""
        processing_class = getattr(self, "processing_class", None)
        if processing_class is None:
            processing_class = getattr(self, "tokenizer", None)

        if processing_class is not None and self.is_world_process_zero():
            processing_class.save_pretrained(output_dir)

    @abstractmethod
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        
        Args:
            model (AbsRerankerModel): The model being trained.
            inputs (dict): A dictionary of input tensors to be passed to the model.
            return_outputs (bool, optional): If ``True``, returns both the loss and the model's outputs. Otherwise,
                returns only the loss. Defaults to ``False``.
        
        Returns:
            Union[torch.Tensor, tuple(torch.Tensor, RerankerOutput)]: The computed loss. If ``return_outputs`` is ``True``, 
                also returns the model's outputs in a tuple ``(loss, outputs)``.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
