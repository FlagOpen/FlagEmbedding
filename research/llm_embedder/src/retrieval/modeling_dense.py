import os
import torch
import faiss
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging
from typing import List, Mapping, Optional, Tuple, Union
from tqdm import tqdm
from .data import RetrievalDataCollator
from ..utils.util import Sequential_Sampler, makedirs, do_nothing

logger = logging.get_logger(__name__)


class DenseRetriever(torch.nn.Module):
    def __init__(self, query_encoder:str='BAAI/bge-base-en', key_encoder:str='BAAI/bge-base-en', pooling_method:List[str]=["cls"], dense_metric:str="cos", query_max_length:int=512, key_max_length:int=512, tie_encoders:bool=True, truncation_side:str="right", dtype:str="fp16", cache_dir:Optional[str]=None, cos_temperature:float=0.01, contrastive_weight:float=0.2, distill_weight:float=1.0, teacher_temperature:float=1.0, student_temperature:float=1.0, negative_cross_device:bool=True, stable_distill:bool=False, accelerator:Accelerator=None, **kwds) -> None:
        super().__init__()
        self.accelerator = accelerator

        self.tie_encoders = tie_encoders
        self.pooling_method = pooling_method
        self.dense_metric = dense_metric
        self.query_max_length = query_max_length
        self.key_max_length = key_max_length
        self.cos_temperature = cos_temperature
        self.contrastive_weight = contrastive_weight
        self.distill_weight = distill_weight
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.negative_cross_device = negative_cross_device and dist.is_initialized()
        self.stable_distill = stable_distill

        logger.info(f"Loading tokenizer and model from {query_encoder}...")

        self.tokenizer = AutoTokenizer.from_pretrained(query_encoder, cache_dir=cache_dir, truncation_side=truncation_side)

        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.query_encoder_name = query_encoder
        self.key_encoder_name = key_encoder
        if tie_encoders:
            encoder = AutoModel.from_pretrained(query_encoder, cache_dir=cache_dir, torch_dtype=dtype).to(self.device)
            self.query_encoder = encoder
            self.key_encoder = encoder
        else:
            self.query_encoder = AutoModel.from_pretrained(query_encoder, cache_dir=cache_dir, torch_dtype=dtype).to(self.device)
            self.key_encoder = AutoModel.from_pretrained(key_encoder, cache_dir=cache_dir, torch_dtype=dtype).to(self.device)

        self.ndim = self.query_encoder.config.hidden_size
        self._index = None
        self._post_init()
        self.eval()

    def _post_init(self):
        """
        1. remove pooler to avoid DDP errors;
        2. remove decoder when necessary
        """
        if hasattr(self.query_encoder, "pooler"):
           self.query_encoder.pooler = None
        if hasattr(self.key_encoder, "pooler"):
           self.key_encoder.pooler = None
        if "dense" in self.pooling_method:
            self.dense_pooler = torch.nn.Linear(self.ndim, self.ndim, bias=False).to(device=self.device, dtype=self.query_encoder.dtype)
            try:
                state_dict = torch.load(os.path.join(self.query_encoder_name, "dense_pooler.bin"), map_location=self.device)
                self.dense_pooler.load_state_dict(state_dict)
            except:
                logger.warning(f"Could not find dense pooler weight in {self.query_encoder_name}, initialize it randomly!")

    def gradient_checkpointing_enable(self):
        self.query_encoder.gradient_checkpointing_enable()
        self.key_encoder.gradient_checkpointing_enable()

    @property
    def device(self):
        if self.accelerator is not None:
            return self.accelerator.device
        else:
            return torch.device("cpu")

    def _gather_tensors(self, local_tensor):
        """
        Gather tensors from all gpus on each process.

        Args:
            local_tensor: the tensor that needs to be gathered

        Returns:
            concatenation of local_tensor in each process
        """
        if local_tensor is None:
            return None
        all_tensors = [torch.empty_like(local_tensor)
                       for _ in range(self.accelerator.num_processes)]
        dist.all_gather(all_tensors, local_tensor.contiguous())
        all_tensors[self.accelerator.process_index] = local_tensor
        return torch.cat(all_tensors, dim=0)

    def _save_to_memmap(self, path: str, shape: tuple, array: np.ndarray, start: int, batch_size: int = 100000):
        """
        Save to numpy array to memmap file.
        """
        if self.accelerator.process_index == 0:
            if os.path.exists(path):
                os.remove(path)
            else:
                makedirs(path)
            memmap = np.memmap(
                path,
                shape=shape,
                mode="w+",
                dtype=array.dtype
            )
            del memmap
        
        self.accelerator.wait_for_everyone()

        logger.info(f"saving array at {path}...")
        memmap = np.memmap(
            path,
            shape=shape,
            mode="r+",
            dtype=array.dtype
        )
        array_length = array.shape[0]
        # add in batch
        end = start + array_length 
        if array_length > batch_size:
            for i in tqdm(range(0, array_length, batch_size), leave=False, ncols=100):
                start_idx = start + i
                end_idx = min(start_idx + batch_size, end)
                memmap[start_idx: end_idx] = array[i: i + (end_idx - start_idx)]
        else:
            memmap[start: end] = array

        self.accelerator.wait_for_everyone()

    def _prepare(self, inputs: Union[str, List[str], Mapping], field="key"):
        """Convert inputs into tokenized input_ids"""
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            if field == "key":
                inputs = self.tokenizer(
                    inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.key_max_length)
                inputs = inputs.to(self.device)
            elif field == "query":
                inputs = self.tokenizer(
                    inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.query_max_length)
                inputs = inputs.to(self.device)
            else:
                raise NotImplementedError
        elif isinstance(inputs, Mapping) and "input_ids" in inputs:
            if field == "key":
                for k, v in inputs.items():
                    inputs[k] = v[:, :self.key_max_length].to(self.device)
            elif field == "query":
                for k, v in inputs.items():
                    inputs[k] = v[:, :self.query_max_length].to(self.device)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Expected inputs of type str, list[str], or dict, got {type(inputs)}!")
        return inputs

    def _pool(self, embeddings, attention_mask):
        if "mean" in self.pooling_method:
            embeddings = embeddings.masked_fill(
                ~attention_mask[..., None].bool(), 0.0)
            embedding = embeddings.sum(
                dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif "cls" in self.pooling_method:
            embedding = embeddings[:, 0]
        elif "decoder" in self.pooling_method:
            embedding = embeddings[:, 0]
        else:
            raise NotImplementedError(
                f"Pooling_method {self.pooling_method} not implemented!")

        if "dense" in self.pooling_method:
            embedding = self.dense_pooler(embedding)
        return embedding

    def encode(self, inputs: Union[str, List[str], Mapping], field:str="key", with_grad:bool=False):
        """Encode inputs into embeddings

        Args:
            inputs: can be string, list of strings, or BatchEncoding results from tokenizer

        Returns:
            Tensor: [batch_size, d_embed]
        """
        if with_grad:
            ctx_manager = do_nothing
        else:
            ctx_manager = torch.no_grad
        
        with ctx_manager():
            inputs = self._prepare(inputs, field=field)

            if field == "key":
                encoder = self.key_encoder
            elif field == "query":
                encoder = self.query_encoder
            else:
                raise ValueError(f"Field {field} not implemented!")

            if hasattr(encoder, "decoder"):
                # AAR uses T5 decoder to produce embedding
                if "decoder" in self.pooling_method:
                    input_ids = inputs['input_ids']
                    bos_token_id = encoder.config.decoder_start_token_id
                    decoder_input_ids = input_ids.new_zeros(input_ids.shape[0], 1) + bos_token_id
                    embeddings = encoder(**inputs, decoder_input_ids=decoder_input_ids).last_hidden_state   # B, 1, D
                else:
                    # only use the encoder part
                    encoder = encoder.encoder
                    embeddings = encoder(**inputs).last_hidden_state    # B, L, D
            else:
                embeddings = encoder(**inputs).last_hidden_state    # B, L, D

            embedding = self._pool(embeddings, inputs["attention_mask"])

            if self.dense_metric == "cos":
                embedding = F.normalize(embedding, p=2, dim=1)
            return embedding

    def _compute_loss(self, query_embedding, key_embedding, teacher_scores):
        if teacher_scores is not None and self.distill_weight > 0:
            do_distill = True
            if self.stable_distill:
                teacher_targets = F.softmax(teacher_scores, dim=-1) # B N
                if self.negative_cross_device:
                    # gather with grad
                    query_embeddings = self._gather_tensors(query_embedding)
                    key_embeddings = self._gather_tensors(key_embedding)
                    teacher_targets = self._gather_tensors(teacher_targets)
                else:
                    query_embeddings = query_embedding
                    key_embeddings = key_embedding
                    teacher_targets = teacher_targets

                scores = query_embeddings.matmul(key_embeddings.transpose(-1, -2))   # B, B * N
                if self.dense_metric == "cos":
                    scores = scores  / self.cos_temperature
                labels = torch.arange(query_embeddings.shape[0], device=self.device)
                labels = labels * (key_embeddings.shape[0] // query_embeddings.shape[0])
                # labels = torch.zeros(query_embeddings.shape[0], device=self.device, dtype=torch.long)
                # scores = 

                distill_loss = 0
                group_size = key_embeddings.shape[0] // query_embeddings.shape[0]
                mask = torch.zeros_like(scores)
                for i in range(group_size):
                    temp_target = labels + i
                    temp_scores = scores + mask
                    loss = F.cross_entropy(temp_scores, temp_target, reduction="none") # B
                    distill_loss = distill_loss + torch.mean(teacher_targets[:, i] * loss)
                    mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1), value=torch.finfo(scores.dtype).min)

            else:
                student_query = query_embedding.unsqueeze(1)    # B, 1, D
                student_key = key_embedding.view(student_query.shape[0], -1, student_query.shape[-1])   # B, N, D
                student_scores = student_query.matmul(student_key.transpose(-1, -2)).squeeze(1)         # B, N
                if self.dense_metric == "cos":
                    student_scores = student_scores  / self.cos_temperature
                student_scores = F.log_softmax(student_scores / self.student_temperature, dim=-1)
                teacher_scores = F.softmax(teacher_scores / self.teacher_temperature, dim=-1)
                distill_loss = F.kl_div(student_scores, teacher_scores, reduction="batchmean")

        else:
            do_distill = False

        if self.contrastive_weight > 0:
            if self.negative_cross_device:
                # gather with grad
                query_embedding = self._gather_tensors(query_embedding)
                key_embedding = self._gather_tensors(key_embedding)
            scores = query_embedding.matmul(key_embedding.transpose(-1, -2))   # B, B * N
            if self.dense_metric == "cos":
                scores = scores  / self.cos_temperature
            # in batch negative
            labels = torch.arange(query_embedding.shape[0], device=self.device)
            labels = labels * (key_embedding.shape[0] // query_embedding.shape[0])
            contrastive_loss = F.cross_entropy(scores, labels)
            do_contrastive = True
        else:
            do_contrastive = False

        if do_distill and do_contrastive:
            loss = contrastive_loss * self.contrastive_weight + distill_loss * self.distill_weight
            # if self.accelerator.process_index == 0:
            #     print(f"distill: {distill_loss * self.distill_weight} contra: {contrastive_loss * self.contrastive_weight} sumup: {loss} contra_weight: {self.contrastive_weight} distill_weight: {self.distill_weight}\n")
        elif do_distill:
            loss = distill_loss
        elif do_contrastive:
            loss = contrastive_loss
        else:
            raise ValueError(f"Neither distill or contrastive learning is enabled!")

        return loss

    def _refresh_config(self, task):
        if hasattr(self, "train_config"):
            # at the first iteration, set default value
            if not hasattr(self, "_contrastive_weight"):
                self._contrastive_weight = self.contrastive_weight
                self._distill_weight = self.distill_weight
                self._teacher_temperature = self.teacher_temperature
                self._student_temperature = self.student_temperature
                self._stable_distill= self.stable_distill

            train_config = self.train_config[task]
            # when there is no setting in the train config, fall back to the default config
            self.contrastive_weight = train_config.get("contrastive_weight", self._contrastive_weight)
            self.distill_weight = train_config.get("distill_weight", self._distill_weight)
            self.teacher_temperature = train_config.get("teacher_temperature", self._teacher_temperature)
            self.student_temperature = train_config.get("student_temperature", self._student_temperature)
            self.stable_distill = train_config.get("stable_distill", self._stable_distill)

    def forward(self, query, key, task, teacher_scores=None, **kwds):
        self._refresh_config(task)

        # batch_size * (1 + nneg), ndim
        key_embedding = self.encode(key, with_grad=True)
        query_embedding = self.encode(query, field="query", with_grad=True)    # batch_size, ndim

        # for debug
        # print(f"************************\n{self.accelerator.process_index}: {query['input_ids'].shape}\n {self.tokenizer.decode(query['input_ids'][0])}\n{self.contrastive_weight}\n{self.distill_weight}\n{teacher_scores[0]}")

        loss = self._compute_loss(query_embedding, key_embedding, teacher_scores)
        # adapted to huggingface trainer
        return {"loss": loss}

    @torch.no_grad()
    def index(self, corpus: Dataset, output_dir="data/outputs", embedding_name=None, index_factory:str="Flat", save_index=False, load_encode=False, save_encode=False, load_index=False, batch_size=500, metric=None, **kwds):
        os.makedirs(output_dir, exist_ok=True)

        if embedding_name is None:
            embedding_name = "embeddings"
        if metric is None:
            metric = self.dense_metric

        encode_path = os.path.join(output_dir, f"{embedding_name}.memmap")
        index_path = os.path.join(output_dir, f"{embedding_name}.{index_factory}.{self.accelerator.process_index}-{self.accelerator.num_processes}.faiss")

        sampler = Sequential_Sampler(len(corpus), self.accelerator.num_processes, self.accelerator.process_index)
        self._corpus_offset = sampler.start

        if load_encode:
            encoded_corpus = np.memmap(
                encode_path,
                mode="r",
                dtype=np.float32
            ).reshape(len(corpus), self.ndim)[sampler.start: sampler.end]

        else:
            # use multiple workers to speed up encoding
            dataloader = DataLoader(
                corpus,
                batch_size=batch_size,
                collate_fn=RetrievalDataCollator(
                    query_max_length=self.query_max_length,
                    key_max_length=self.key_max_length,
                    tokenizer=self.tokenizer,
                ),
                sampler=sampler,
                pin_memory=True,
                num_workers=8,
            )

            offset = 0
            encoded_corpus = np.zeros((len(sampler), self.ndim), dtype=np.float32)

            for step, inputs in enumerate(tqdm(dataloader, desc="Indexing")):
                embeddings = self.encode(inputs["content"])   # batch_size, ndim
                # NOTE: we cannot use non_blocking here, otherwise nothing can be saved
                encoded_corpus[offset: offset + embeddings.shape[0]] = embeddings.cpu().numpy()
                offset += embeddings.shape[0]
                # if step > 10:
                #     break

            if save_encode:
                self._save_to_memmap(
                    encode_path,
                    shape=(len(corpus), self.ndim),
                    array=encoded_corpus,
                    start=sampler.start
                )            

        index = FaissIndex(self.device)
        if load_index:
            index.load(index_path)
        else:
            index.build(encoded_corpus, index_factory, metric)
        
        if save_index:
            index.save(index_path)

        self._index = index
        self.accelerator.wait_for_everyone()
        return encoded_corpus

    @torch.no_grad()
    def search(self, inputs: Union[str, List[str], Mapping], hits:int=10, **kwds):
        assert self._index is not None, "Make sure there is an indexed corpus!"

        all_scores = []
        all_indices = []

        embeddings = self.encode(inputs, field="query").cpu().numpy().astype(np.float32, order="C")
        batch_scores, batch_indices = self._index.search(embeddings, hits)
        # offset
        batch_indices += self._corpus_offset

        # gather and merge results from all processes
        # move to cpu for faster sorting and merging
        if self.accelerator.num_processes > 1:
            batch_scores = torch.as_tensor(batch_scores, device=self.device)
            batch_indices = torch.as_tensor(batch_indices, device=self.device)
            gathered_batch_scores = self.accelerator.gather(batch_scores).unflatten(0, (self.accelerator.num_processes, -1)).tolist()
            gathered_batch_indices = self.accelerator.gather(batch_indices).unflatten(0, (self.accelerator.num_processes, -1)).tolist()
        else:
            gathered_batch_scores = batch_scores[None, ...].tolist()
            gathered_batch_indices = batch_indices[None, ...].tolist()

        for batch_idx in range(batch_scores.shape[0]):
            score = sum([gathered_batch_scores[i][batch_idx] for i in range(self.accelerator.num_processes)], [])
            indice = sum([gathered_batch_indices[i][batch_idx] for i in range(self.accelerator.num_processes)], [])
            # take care of -1s, which may be returned by faiss
            pair = sorted(zip(score, indice), key=lambda x: x[0] if x[1] >= 0 else -float('inf'), reverse=True)[:hits]
            all_scores.append([x[0] for x in pair])
            all_indices.append([x[1] for x in pair])

        all_scores = np.array(all_scores, dtype=np.float32)
        all_indices = np.array(all_indices)
        return all_scores, all_indices
    
    @torch.no_grad()
    def rerank(self, query, key, key_mask=None, **kwds):
        query_embeddings = self.encode(query, field="query")
        key_embeddings = self.encode(key)
        key_embeddings = key_embeddings.unflatten(0, (query_embeddings.shape[0], -1))   # batch_size, key_num, embedding_dim
        score = torch.einsum("bnd,bd->bn", key_embeddings, query_embeddings)    # batch_size, key_num
        # mask padded candidates
        if key_mask is not None:
            score = score.masked_fill(~key_mask.bool(), torch.finfo(key_embeddings.dtype).min)

        score, indice = score.sort(dim=-1, descending=True)
        # NOTE: set the indice to -1 so that this prediction is ignored when computing metrics
        indice[score == torch.finfo(score.dtype).min] = -1
        return score, indice

    def save_pretrained(self, output_dir: str, *args, **kwargs):
        if self.tie_encoders:
            self.tokenizer.save_pretrained(
                os.path.join(output_dir, "encoder"))
            self.query_encoder.save_pretrained(
                os.path.join(output_dir, "encoder"))
            if hasattr(self, "dense_pooler"):
                torch.save(self.dense_pooler.state_dict(), os.path.join(output_dir, "encoder", "dense_pooler.bin"))

        else:
            self.tokenizer.save_pretrained(
                os.path.join(output_dir, "query_encoder"))
            self.query_encoder.save_pretrained(
                os.path.join(output_dir, "query_encoder"))
            self.key_tokenizer.save_pretrained(
                os.path.join(output_dir, "key_encoder"))
            self.key_encoder.save_pretrained(
                os.path.join(output_dir, "key_encoder"))
            if hasattr(self, "dense_pooler"):
                torch.save(self.dense_pooler.state_dict(), os.path.join(output_dir, "query_encoder", "dense_pooler.bin"))


class FaissIndex:
    def __init__(self, device) -> None:
        if isinstance(device, torch.device):
            if device.index is None:
                device = "cpu"
            else:
                device = device.index
        self.device = device

    def build(self, encoded_corpus, index_factory, metric):
        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError(f"Metric {metric} not implemented!")
        
        index = faiss.index_factory(encoded_corpus.shape[1], index_factory, metric)
        
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            # logger.info("using fp16 on GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)

        logger.info("training index...")
        index.train(encoded_corpus)
        logger.info("adding embeddings...")
        index.add(encoded_corpus)
        self.index = index
        return index

    def load(self, index_path):
        logger.info(f"loading index from {index_path}...")
        index = faiss.read_index(index_path)
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)
        self.index = index
        return index

    def save(self, index_path):
        logger.info(f"saving index at {index_path}...")
        if isinstance(self.index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, query, hits):
        return self.index.search(query, k=hits)
