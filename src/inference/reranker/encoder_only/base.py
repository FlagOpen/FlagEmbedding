import torch
import numpy as np
from tqdm import tqdm
from typing import Any, List, Union, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer, is_torch_npu_available

from src.abc.inference import AbsReranker

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BaseReranker(AbsReranker):
    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = None,
        device: Union[str, int] = None, # specify device, such as "cuda:0" or "0"
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            use_fp16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

        self.kwargs = kwargs
        
        if device and isinstance(device, str):
            self.device = torch.device(device)
            self.num_gpus = 1
            if device == 'cpu':
                use_fp16 = False
        else:
            if torch.cuda.is_available():
                if device is not None:
                    self.device = torch.device(f"cuda:{device}")
                else:
                    self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        
        if self.use_fp16: self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)
    
    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 256,
                      max_length: int = 512,
                      normalize: bool = False,
                      **kwargs: Any) -> List[float]:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        sentence_pairs_length = []
        for i in range(0, len(sentence_pairs), 1024):
            start, end = i, min(i + 1024, len(sentence_pairs))
            tmp_inputs = self.tokenizer(sentence_pairs[start: end])['input_ids']
            sentence_pairs_length.extend(-len(s) for s in tmp_inputs)
        length_sorted_idx = np.argsort(sentence_pairs_length)
        sentences_pairs_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        flag = False
        while flag is False:
            test_sentences_batch = sentences_pairs_sorted[: min(batch_size, len(sentence_pairs))]
            try:
                inputs = self.tokenizer(
                    test_sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=max_length,
                    **kwargs
                ).to(self.device)
                scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4

        all_scores = []
        for start_index in tqdm(range(0, len(sentences_pairs_sorted), batch_size), desc="Compute Scores",
                                disable=len(sentences_pairs_sorted) < 128):
            sentences_batch = sentences_pairs_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                **kwargs
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores