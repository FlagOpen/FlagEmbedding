import logging
from tqdm import tqdm
from typing import List, Optional

from FlagEmbedding.abc.evaluation import AbsEvalDataLoader

logger = logging.getLogger(__name__)


class CustomEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return []

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        return ["test"]
