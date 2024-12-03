import torch
import warnings
import numpy as np
from tqdm import tqdm, trange
from typing import Any, List, Union, Tuple, Optional
from peft import PeftModel
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from FlagEmbedding.abc.inference import AbsReranker
from FlagEmbedding.inference.reranker.encoder_only.base import sigmoid
from FlagEmbedding.inference.reranker.decoder_only.base import DatasetForReranker, Collater

from .models.modeling_minicpm_reranker import LayerWiseMiniCPMForCausalLM


def last_logit_pool_layerwise(logits: Tensor,
                              attention_mask: Tensor) -> Tensor:
    """Pool the last logit.

    Args:
        logits (torch.Tensor): The output logits of the model.
        attention_mask (torch.Tensor): Attention mask.

    Returns:
        torch.Tensor: The tensor after pooling.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]


class LayerWiseLLMReranker(AbsReranker):
    """Base reranker class for layerwise LLM like decoder only models.

    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        peft_path (Optional[str], optional): Path to the PEFT config. Defaults to :data:`None`.
        use_fp16 (bool, optional): If true, use half-precision floating-point to speed up computation with a slight performance 
            degradation. Defaults to :data:`False`. Defaults to :data:`False`.
        use_bf16 (bool, optional): Another type of half-precision floating-point, you can use bf16 if the hardware supports. 
            Defaults to :data:False.
        query_instruction_for_rerank (str, optional): Query instruction for retrieval tasks, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`"A: "`.
        query_instruction_format (str, optional): The template for :attr:`query_instruction_for_rerank`. Defaults to :data:`"{}{}"`.
        passage_instruction_for_rerank (str, optional): Passage instruction for retrieval tasks, which will be used with
            with :attr:`passage_instruction_format`. Defaults to :data:`"B: "`.
        passage_instruction_format (str, optional): The template for passage. Defaults to "{}{}".
        cache_dir (Optional[str], optional): Cache directory for the model. Defaults to :data:`None`.
        trust_remote_code (bool, optional): trust_remote_code. Defaults to :data:`False`.
        devices (Union[str, List[str], List[int]], optional): Devices to use for model inference, such as ["cuda:0"] or ["0"].
            Defaults to :data:`None`.
        cutoff_layers (Optional[List[int]]): Pick which layers are used for computing the score. Defaults to :data:`None`.
        prompt (Optional[str], optional): Prompt for the specific task. Defaults to :data:`None`.
        batch_size (int, optional): Batch size for inference. Defaults to :data:`128`.
        query_max_length (int, optional): Maximum length for queries. If not specified, will be 3/4 of :attr:`max_length`.
            Defaults to :data:`None`.
        max_length (int, optional): Maximum length of passages. Defaults to :data`512`.
        normalize (bool, optional): If True, use Sigmoid to normalize the results. Defaults to :data:`False`.
    """
    def __init__(
        self,
        model_name_or_path: str,
        peft_path: Optional[str] = None,
        use_fp16: bool = False,
        use_bf16: bool = False,
        query_instruction_for_rerank: str = "A: ",
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_rerank
        passage_instruction_for_rerank: str = "B: ",
        passage_instruction_format: str = "{}{}", # specify the format of passage_instruction_for_rerank
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        devices: Optional[Union[str, List[str], List[int]]] = None, # specify devices, such as ["cuda:0"] or ["0"]
        # inference
        cutoff_layers: Optional[List[int]] = None, 
        prompt: Optional[str] = None,
        batch_size: int = 128,
        query_max_length: Optional[int] = None,
        max_length: int = 512,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            query_instruction_for_rerank=query_instruction_for_rerank,
            query_instruction_format=query_instruction_format,
            passage_instruction_for_rerank=passage_instruction_for_rerank,
            passage_instruction_format=passage_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            max_length=max_length,
            normalize=normalize,
            **kwargs
        )

        self.cutoff_layers = cutoff_layers
        self.prompt = prompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )

        if use_bf16 is False and use_fp16 is False:
            warnings.warn("Due to model constraints, `use_bf16` and `use_fp16` cannot both be `False`. Here, `use_fp16` is set to `True` by default.", UserWarning)
            self.use_fp16 = True
        
        try:
            self.model = LayerWiseMiniCPMForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
            )
        except:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
            )
        if peft_path:
            self.model = PeftModel.from_pretrained(self.model,peft_path)
            self.model = self.model.merge_and_unload()

    @torch.no_grad()
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: Optional[int] = None,
        query_max_length: Optional[int] = None,
        max_length: Optional[int] = None,
        cutoff_layers: Optional[List[int]] = None, 
        prompt: Optional[str] = None,
        normalize: Optional[bool] = None,
        use_dataloader: bool = False,
        num_workers: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs: Any
    ) -> List[float]:
        """Compute the relevance scores using a single GPU.

        Args:
            sentence_pairs (Union[List[Tuple[str, str]], Tuple[str, str]]): Input sentence pairs to compute scores.
            batch_size (Optional[int], optional): Number of inputs for each iter. Defaults to :data:`None`.
            query_max_length (Optional[int], optional): Maximum length of tokens of queries. Defaults to :data:`None`.
            max_length (Optional[int], optional): Maximum length of tokens. Defaults to :data:`None`.
            cutoff_layers (Optional[List[int]], optional): Pick which layers are used for computing the score. Defaults to :data:`None`.
            prompt (Optional[str], optional): Prompt for the specific task. Defaults to :data:`None`.
            normalize (Optional[bool], optional): If True, use Sigmoid to normalize the results. Defaults to :data:`None`.
            use_dataloader (bool, optional): If True, will use the dataloader to load the datasets. Defaults to :data:`False`.
            num_workers (int, optional): Number of workers for dataloader. Defaults to :data:`None`.
            device (Optional[str], optional): Device to use for computation. Defaults to :data:`None`.

        Returns:
            List[float]: The computed scores.
        """
        if cutoff_layers is None: cutoff_layers = self.cutoff_layers
        if prompt is None: prompt = self.prompt
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.max_length
        if query_max_length is None:
            if self.query_max_length is not None:
                query_max_length = self.query_max_length
            else:
                query_max_length = max_length * 3 // 4
        if normalize is None: normalize = self.normalize

        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        # tokenize without padding to get the correct length
        all_queries_inputs = []
        all_passages_inputs = []
        for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize",
                                  disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
                **kwargs
            )
            passages_inputs_batch = self.tokenizer(
                passages,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                **kwargs
            )
            queries_inputs_batch = [{
                k: queries_inputs_batch[k][i] for k in queries_inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            passages_inputs_batch = [{
                k: passages_inputs_batch[k][i] for k in passages_inputs_batch.keys()
            } for i in range(len(sentences_batch))]

            all_queries_inputs.extend(queries_inputs_batch)
            all_passages_inputs.extend(passages_inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) - len(y['input_ids']) for (x, y) in zip(all_queries_inputs, all_passages_inputs)])
        all_queries_inputs_sorted = [all_queries_inputs[i] for i in length_sorted_idx]
        all_passages_inputs_sorted = [all_passages_inputs[i] for i in length_sorted_idx]

        # other inputs
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors=None,
            add_special_tokens=False
        )['input_ids']
        sep = "\n"
        sep_inputs = self.tokenizer(
            sep,
            return_tensors=None,
            add_special_tokens=False
        )['input_ids']
        encode_max_length = max_length + len(sep_inputs) + len(prompt_inputs)

        # adjust batch size
        flag = False
        while flag is False:
            try:
                batch_inputs = []
                for query_inputs, passage_inputs in zip(
                    all_queries_inputs_sorted[:min(len(all_queries_inputs_sorted), batch_size)], 
                    all_passages_inputs_sorted[:min(len(all_passages_inputs_sorted), batch_size)]
                ):
                    item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                        sep_inputs + passage_inputs['input_ids'],
                        truncation='only_second',
                        max_length=encode_max_length,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False
                    )
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
                    if 'position_ids' in item.keys():
                        item['position_ids'] = list(range(len(item['input_ids'])))
                    batch_inputs.append(item)

                collater_instance = Collater(self.tokenizer, encode_max_length)
                batch_inputs = collater_instance([{
                        'input_ids': item['input_ids'],
                        'attention_mask': item['attention_mask']
                    } for item in batch_inputs]
                )

                batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}

                self.model(**batch_inputs, output_hidden_states=True, cutoff_layers=cutoff_layers)
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
            except torch.OutofMemoryError as e:
                batch_size = batch_size * 3 // 4

        dataset, dataloader = None, None
        if use_dataloader:
            if num_workers is None:
                num_workers = min(batch_size, 16)
            dataset = DatasetForReranker(
                all_queries_inputs_sorted,
                all_passages_inputs_sorted,
                self.model_name_or_path,
                max_length,
                cache_dir=self.cache_dir,
                prompt=prompt,
                **kwargs
            )
            dataloader = DataLoader(
                dataset, shuffle=False, batch_size=batch_size, drop_last=False,
                num_workers=num_workers,
                collate_fn=Collater(self.tokenizer, encode_max_length)
            )

        all_scores = []
        if dataloader is not None:
            for inputs in tqdm(dataloader):
                inputs = inputs.to(device)

                outputs = self.model(**inputs, output_hidden_states=True, cutoff_layers=cutoff_layers)
                all_logits = outputs.logits
                tmp_all_scores = []
                for logits in all_logits:
                    scores = last_logit_pool_layerwise(logits, inputs['attention_mask'])
                    tmp_all_scores.append(scores.contiguous())

                if len(all_scores) == 0:
                    for _ in range(len(tmp_all_scores)):
                        all_scores.append([])

                for i in range(len(tmp_all_scores)):
                    all_scores[i].extend(tmp_all_scores[i].cpu().float().tolist())
        else:
            for batch_start in trange(0, len(all_queries_inputs_sorted), batch_size):
                queries_inputs = all_queries_inputs_sorted[batch_start:batch_start+batch_size]
                passages_inputs = all_passages_inputs_sorted[batch_start:batch_start+batch_size]

                batch_inputs = []
                for query_inputs, passage_inputs in zip(queries_inputs, passages_inputs):
                    item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                        sep_inputs + passage_inputs['input_ids'],
                        truncation='only_second',
                        max_length=encode_max_length,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False
                    )
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
                    if 'position_ids' in item.keys():
                        item['position_ids'] = list(range(len(item['input_ids'])))
                    batch_inputs.append(item)

                collater_instance = Collater(self.tokenizer, encode_max_length)
                batch_inputs = collater_instance([{
                    'input_ids': item['input_ids'],
                    'attention_mask': item['attention_mask']
                    } for item in batch_inputs]
                )

                batch_inputs = {key: val.to(device) for key, val in batch_inputs.items()}

                outputs = self.model(**batch_inputs, output_hidden_states=True, cutoff_layers=cutoff_layers)
                all_logits = outputs.logits
                tmp_all_scores = []
                for logits in all_logits:
                    scores = last_logit_pool_layerwise(logits, batch_inputs['attention_mask'])
                    tmp_all_scores.append(scores.contiguous())

                if len(all_scores) == 0:
                    for _ in range(len(tmp_all_scores)):
                        all_scores.append([])

                for i in range(len(tmp_all_scores)):
                    all_scores[i].extend(tmp_all_scores[i].cpu().float().tolist())

        for i in range(len(all_scores)):
            all_scores[i] = [all_scores[i][idx] for idx in np.argsort(length_sorted_idx)]
            if normalize:
                all_scores[i] = [sigmoid(score) for score in all_scores[i]]
        
        if len(all_scores) == 1 and isinstance(all_scores[0], list):
            all_scores = all_scores[0]
            
        return all_scores
