import os
import torch
import logging
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer
from FlagEmbedding.finetune.embedder.encoder_only.base.metrics import mapk, apk, mean_average_precision_at_k

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderTrainer(AbsEmbedderTrainer):
    """
    Trainer class for base encoder models.
    """
    def __init__(self, corpus, eval_data_collator, corpus_collator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corpus = corpus
        self.eval_data_collator = eval_data_collator
        self.corpus_collator = corpus_collator
    
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

        # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     save_ckpt_for_sentence_transformers(output_dir,
        #                                         pooling_mode=self.args.sentence_pooling_method,
        #                                         normlized=self.args.normlized)
    
    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        self.model.eval()
        
        if eval_dataset is None and self.eval_dataset is None:
            logger.warning("No evaluation dataset provided. Skipping evaluation.")
            return
            
        corpus_dataloader = DataLoader(
            self.corpus,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            collate_fn=self.corpus_collator,
        )
        
        def batch_to_device(batch, target_device):
            """
            send a pytorch batch to a device (CPU/GPU)
            """
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(target_device)
            return batch
        
        corpus_embeddings = []
        for batch in tqdm(corpus_dataloader, desc="Encoding corpus"):
            batch = batch_to_device(batch["text_inputs"], self.accelerator.device)
            embeddings = self.model.encode(batch)
            embeddings = embeddings.detach().cpu().numpy()
            corpus_embeddings.append(embeddings)
        corpus_embeddings = np.concatenate(corpus_embeddings, axis=0)
        
        index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            collate_fn=self.eval_data_collator,
        )
        query_embeddings = []
        correct_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            correct_id = batch["correct_id"]
            batch = batch_to_device(batch["text_inputs"], self.accelerator.device)
            embeddings = self.model.encode(batch)
            embeddings = embeddings.detach().cpu().numpy()
            query_embeddings.append(embeddings)
            correct_ids.extend(correct_id)
        query_embeddings = np.concatenate(query_embeddings, axis=0)
        distances, indices = index.search(query_embeddings, k=25)
        mapk_score = mean_average_precision_at_k(correct_ids, indices, 25)
        logger.info(f"mapk_score: {mapk_score}")
        
        self._memory_tracker.stop_and_update_metrics()

        return mapk_score
