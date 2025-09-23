import os
import logging
from typing import Optional

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderTrainer(AbsEmbedderTrainer):
    """
    Trainer class for base encoder models.
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model to directory.

        Args:
            output_dir (Optional[str], optional): Output directory to save the model. Defaults to ``None``.

        Raises:
            NotImplementedError
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.is_world_process_zero():
            self.save_ckpt_for_sentence_transformers(output_dir,
                                                    pooling_mode=self.args.sentence_pooling_method)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)