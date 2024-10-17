import torch
import numpy as np
from tqdm import tqdm
from typing import Any, List, Union, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer, is_torch_npu_available

from FlagEmbedding.abc.inference import AbsReranker


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
                    self.num_gpus = 1
                else:
                    self.device = torch.device("cuda")
                    self.num_gpus = torch.cuda.device_count()
            else:
                self.num_gpus = -1  # TODO: DataParallel for other devices
                if torch.backends.mps.is_available():
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
        
        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentence_pairs))]
            all_inputs.extend(inputs_batch)
        
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        
        # adjust batch size
        flag = False
        while flag is False:
            try:
                test_inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[:min(len(all_inputs_sorted), batch_size)],
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt',
                    **kwargs
                ).to(self.device)
                scores = self.model(**test_inputs_batch, return_dict=True).logits.view(-1, ).float()
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4

        all_scores = []
        for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size), desc="Compute Scores",
                                disable=len(all_inputs_sorted) < 128):
            sentences_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer.pad(
                sentences_batch,
                padding=True,
                max_length=max_length,
                return_tensors='pt',
                **kwargs
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores
