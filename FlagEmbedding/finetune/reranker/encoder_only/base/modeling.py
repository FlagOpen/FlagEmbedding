from transformers import PreTrainedModel, AutoTokenizer
import logging

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel

logger = logging.getLogger(__name__)


class CrossEncoderModel(AbsRerankerModel):
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: AutoTokenizer = None,
        train_batch_size: int = 4,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=train_batch_size,
        )

    def encode(self, features):
        return self.model(**features, return_dict=True).logits
