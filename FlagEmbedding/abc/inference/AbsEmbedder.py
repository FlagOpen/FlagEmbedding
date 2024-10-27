import logging
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
from typing import Any, Union, List, Dict, Literal, Optional

import queue
import multiprocessing as mp
from multiprocessing import Queue

import math
import torch
import numpy as np
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)


class AbsEmbedder(ABC):
    """
    Base class for embedder.
    Extend this class and implement `encode_queries`, `encode_passages`, `encode` for custom embedders.
    """
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: str = None,
        query_instruction_format: str = "{}{}", # specify the format of query_instruction_for_retrieval
        devices: Union[str, int, List[str], List[int]] = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        instruction: str = None,
        instruction_format: str = "{}{}",
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ):
        self.model_name_or_path = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.target_devices = self.get_target_devices(devices)

        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.instruction = instruction
        self.instruction_format = instruction_format
        self.convert_to_numpy = convert_to_numpy

        for k in kwargs:
            setattr(self, k, kwargs[k])

        self.kwargs = kwargs

        # tokenizer and model are initialized in the child class
        self.tokenizer = None
        self.model = None

    @staticmethod
    def get_target_devices(devices: Union[str, int, List[str], List[int]]) -> List[str]:
        if devices is None:
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                return [f"npu:{i}" for i in range(torch.npu.device_count())]
            elif torch.backends.mps.is_available():
                return [f"mps:{i}" for i in range(torch.mps.device_count())]
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

    @staticmethod
    def get_detailed_instruct(instruction_format: str, instruction: str, sentence: str):
        return instruction_format.format(instruction, sentence)

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.query_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=self.query_instruction_for_retrieval,
            instruction_format=self.query_instruction_format,
            **kwargs
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ):
        passage_instruction_for_retrieval = self.kwargs.get("passage_instruction_for_retrieval", None)
        passage_instruction_format = self.kwargs.get("passage_instruction_format", "{}{}")

        if batch_size is None: batch_size = self.batch_size
        if max_length is None: max_length = self.passage_max_length
        if convert_to_numpy is None: convert_to_numpy = self.convert_to_numpy

        return self.encode(
            corpus,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            instruction=passage_instruction_for_retrieval,
            instruction_format=passage_instruction_format,
            **kwargs
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        instruction: Optional[str] = None,
        instruction_format: Optional[str] = None,
        **kwargs: Any
    ):
        if instruction is None: instruction = self.instruction
        if instruction_format is None: instruction_format = self.instruction_format

        if instruction is not None:
            if isinstance(sentences, str):
                sentences = self.get_detailed_instruct(instruction_format, instruction, sentences)
            else:
                sentences = [self.get_detailed_instruct(instruction_format, instruction, sentence) for sentence in sentences]

        if isinstance(sentences, str) or len(self.target_devices) == 1:
            return self.encode_single_device(
                sentences,
                batch_size=batch_size,
                max_length=max_length,
                convert_to_numpy=convert_to_numpy,
                device=self.target_devices[0],
                **kwargs
            )

        pool = self.start_multi_process_pool(AbsEmbedder._encode_multi_process_worker)
        embeddings = self.encode_multi_process(
            sentences,
            pool,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )
        self.stop_multi_process_pool(pool)
        return embeddings

    @abstractmethod
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: str = None,
        **kwargs: Any,
    ):
        """
        This method should encode sentences and return embeddings on a single device.
        """
        pass

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L807
    def start_multi_process_pool(
        self,
        process_target_func: Any,
    ) -> Dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, self.target_devices))))

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in tqdm(self.target_devices, desc='initial target device'):
            p = ctx.Process(
                target=process_target_func,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L976
    @staticmethod
    def _encode_multi_process_worker(
        target_device: str, model: 'AbsEmbedder', input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, sentences, kwargs = (
                    input_queue.get()
                )
                embeddings = model.encode_single_device(
                    sentences,
                    device=target_device,
                    **kwargs
                )

                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
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

    # adapted from https://github.com/UKPLab/sentence-transformers/blob/1802076d4eae42ff0a5629e1b04e75785d4e193b/sentence_transformers/SentenceTransformer.py#L877
    def encode_multi_process(
        self,
        sentences: List[str],
        pool: Dict[Literal["input", "output", "processes"], Any],
        **kwargs
    ):
        chunk_size = math.ceil(len(sentences) / len(pool["processes"]))

        input_queue = pool["input"]
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
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
        embeddings = self._concatenate_results_from_multi_process([result[1] for result in results_list])
        return embeddings

    def _concatenate_results_from_multi_process(self, results_list: List[Union[torch.Tensor, np.ndarray, Any]]):
        if isinstance(results_list[0], torch.Tensor):
            return torch.cat(results_list, dim=0)
        elif isinstance(results_list[0], np.ndarray):
            return np.concatenate(results_list, axis=0)
        else:
            raise NotImplementedError("Unsupported type for results_list")