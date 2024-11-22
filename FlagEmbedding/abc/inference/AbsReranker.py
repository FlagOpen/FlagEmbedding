import logging
from abc import ABC, abstractmethod
from typing import Any, Union, List, Tuple, Dict, Literal, Optional

import multiprocessing as mp
from multiprocessing import Queue

import math
import gc
import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)


class AbsReranker(ABC):
    """
    Base class for Reranker.
    Extend this class and implement :meth:`compute_score_single_gpu` for custom rerankers.

    Args:
        model_name_or_path (str): If it's a path to a local model, it loads the model from the path. Otherwise tries to download and
            load a model from HuggingFace Hub with the name.
        use_fp16 (bool, optional): If true, use half-precision floating-point to speed up computation with a slight performance 
            degradation. Defaults to :data:`False`.
        query_instruction_for_rerank: (Optional[str], optional): Query instruction for reranking, which will be used with
            with :attr:`query_instruction_format`. Defaults to :data:`None`.
        query_instruction_format: (str, optional): The template for :attr:`query_instruction_for_rerank`. Defaults to :data:`"{}{}"`.
        passage_instruction_for_rerank (Optional[str], optional): Passage instruction for reranking. Defaults to :data:`None`.
        passage_instruction_format (str, optional): Passage instruction format when using :attr:`passage_instruction_for_rerank`. 
            Defaults to :data:`"{}{}"`.
        devices (Optional[Union[str, int, List[str], List[int]]], optional): Devices to use for model inference. Defaults to :data:`None`.
        batch_size (int, optional): Batch size for inference. Defaults to :data:`128`.
        query_max_length (int, optional): Maximum length for query. Defaults to :data:`None`.
        max_length (int, optional): Maximum length. Defaults to :data:`512`.
        normalize (bool, optional): If true, normalize the result. Defaults to :data:`False`.
        kwargs (Dict[Any], optional): Additional parameters for HuggingFace Transformers config or children classes.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_fp16: bool = False,
        query_instruction_for_rerank: Optional[str] = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_rerank
        passage_instruction_for_rerank: Optional[str] = None,
        passage_instruction_format: str = "{}{}", # specify the format of passage_instruction_for_rerank
        devices: Optional[Union[str, int, List[str], List[int]]] = None,
        # inference
        batch_size: int = 128,
        query_max_length: Optional[int] = None,
        max_length: int = 512,
        normalize: bool = False,
        **kwargs: Any,
    ):
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
        self.query_instruction_for_rerank = query_instruction_for_rerank
        self.query_instruction_format = query_instruction_format
        self.passage_instruction_for_rerank = passage_instruction_for_rerank
        self.passage_instruction_format = passage_instruction_format
        self.target_devices = self.get_target_devices(devices)
        
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.max_length = max_length
        self.normalize = normalize

        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.kwargs = kwargs

        # tokenizer and model are initialized in the child class
        self.model = None
        self.tokenizer = None
        self.pool = None

    def stop_self_pool(self):
        if self.pool is not None:
            self.stop_multi_process_pool(self.pool)
            self.pool = None
        try:
            self.model.to('cpu')
            torch.cuda.empty_cache()
        except:
            pass
        gc.collect()

    @staticmethod
    def get_target_devices(devices: Union[str, int, List[str], List[int]]) -> List[str]:
        """

        Args:
            devices (Union[str, int, List[str], List[int]]): Specified devices, can be `str`, `int`, list of `str`, or list of `int`.

        Raises:
            ValueError: Devices should be a string or an integer or a list of strings or a list of integers.

        Returns:
            List[str]: A list of target devices in format
        """
        if devices is None:
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                return [f"npu:{i}" for i in range(torch.npu.device_count())]
            elif torch.backends.mps.is_available():
                return ["mps"]
            else:
                return ["cpu"]
        elif isinstance(devices, str):
            return [devices]
        elif isinstance(devices, int):
            return [f"cuda:{devices}"]
        elif isinstance(devices, list):
            if isinstance(devices[0], str):
                return devices
            elif isinstance(devices[0], int):
                return [f"cuda:{device}" for device in devices]
            else:
                raise ValueError("devices should be a string or an integer or a list of strings or a list of integers.")
        else:
            raise ValueError("devices should be a string or an integer or a list of strings or a list of integers.")

    def get_detailed_instruct(self, instruction_format: str, instruction: str, sentence: str):
        """Combine the instruction and sentence along with the instruction format.

        Args:
            instruction_format (str): Format for instruction.
            instruction (str): The text of instruction.
            sentence (str): The sentence to concatenate with.

        Returns:
            str: The complete sentence with instruction
        """
        return instruction_format.format(instruction, sentence)
    
    def get_detailed_inputs(self, sentence_pairs: Union[str, List[str]]):
        """get detailed instruct for all the inputs

        Args:
            sentence_pairs (Union[str, List[str]]): Input sentence pairs

        Returns:
            list[list[str]]: The complete sentence pairs with instruction
        """
        if isinstance(sentence_pairs, str):
            sentence_pairs = [sentence_pairs]

        if self.query_instruction_for_rerank is not None:
            if self.passage_instruction_for_rerank is None:
                return [
                    [
                        self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_rerank, sentence_pair[0]),
                        sentence_pair[1]
                    ] for sentence_pair in sentence_pairs
                ]
            else:
                return [
                    [
                        self.get_detailed_instruct(self.query_instruction_format, self.query_instruction_for_rerank, sentence_pair[0]),
                        self.get_detailed_instruct(self.passage_instruction_format, self.passage_instruction_for_rerank, sentence_pair[1])
                    ] for sentence_pair in sentence_pairs
                ]
        else:
            if self.passage_instruction_for_rerank is None:
                return [
                    [
                        sentence_pair[0],
                        sentence_pair[1]
                    ] for sentence_pair in sentence_pairs
                ]
            else:
                return [
                    [
                        sentence_pair[0],
                        self.get_detailed_instruct(self.passage_instruction_format, self.passage_instruction_for_rerank, sentence_pair[1])
                    ] for sentence_pair in sentence_pairs
                ]

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        **kwargs
    ):
        """Compute score for each sentence pair

        Args:
            sentence_pairs (Union[List[Tuple[str, str]], Tuple[str, str]]): Input sentence pairs to compute.

        Returns:
            numpy.ndarray: scores of all the sentence pairs.
        """
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        sentence_pairs = self.get_detailed_inputs(sentence_pairs)

        if isinstance(sentence_pairs, str) or len(self.target_devices) == 1:
            return self.compute_score_single_gpu(
                sentence_pairs,
                device=self.target_devices[0],
                **kwargs
            )

        if self.pool is None:
            self.pool = self.start_multi_process_pool()
        scores = self.encode_multi_process(sentence_pairs,
                                           self.pool,
                                           **kwargs)
        return scores

    def __del__(self):
        self.stop_self_pool()

    @abstractmethod
    def compute_score_single_gpu(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        query_max_length: Optional[int] = None,
        max_length: int = 512,
        normalize: bool = False,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        This method should compute the scores of sentence_pair and return scores.
        """
        pass

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    def start_multi_process_pool(self) -> Dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, self.target_devices))))

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in tqdm(self.target_devices, desc='initial target device'):
            p = ctx.Process(
                target=AbsReranker._encode_multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    def encode_multi_process(
        self,
        sentence_pairs: List,
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs
    ) -> np.ndarray:
        chunk_size = math.ceil(len(sentence_pairs) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence_pair in sentence_pairs:
            chunk.append(sentence_pair)
            if len(chunk) >= chunk_size:
                input_queue.put(
                    [last_chunk_id, chunk, kwargs]
                )
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks")],
            key=lambda x: x[0],
        )
        scores = np.concatenate([result[1] for result in results_list])
        return scores

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    @staticmethod
    def _encode_multi_process_worker(
            target_device: str, model: 'AbsReranker', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.compute_score_single_gpu(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except:
                break

    # copied from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L857
    @staticmethod
    def stop_multi_process_pool(pool: Dict[Literal["input", "output", "processes"], Any]) -> None:
        """
        Stops all processes started with start_multi_process_pool.

        Args:
            pool (Dict[str, object]): A dictionary containing the input queue, output queue, and process list.

        Returns:
            None
        """
        for p in pool["processes"]:
            p.terminate()

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()
