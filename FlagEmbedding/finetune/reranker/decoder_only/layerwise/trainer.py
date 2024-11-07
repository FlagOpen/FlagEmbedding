import os
import torch
import logging
from typing import Optional
# from transformers.deepspeed import is_deepspeed_zero3_enabled
from peft import get_peft_model_state_dict

from FlagEmbedding.abc.finetune.reranker import AbsRerankerTrainer

logger = logging.getLogger(__name__)


class DecoderOnlyRerankerTrainer(AbsRerankerTrainer):
    """
    Trainer class for encoder only base reranker models.
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

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # if is_deepspeed_zero3_enabled():
        #     if state_dict is None:
        #         state_dict = self.model.state_dict()
        #     prefix = 'model.'
        #     assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        #     state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        #     lora_state_dict = get_peft_model_state_dict(self.model.model, state_dict)
        #     if self.args.process_index <= 0:
        #         torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
        #         print(f"Save adapter model at {output_dir}")
