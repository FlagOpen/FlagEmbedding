import torch
import math
import logging
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from transformers.modeling_utils import ModelOutput
from .modeling_lm import LM
from ..utils.util import save_pickle, load_pickle

logger = logging.getLogger(__name__)


@dataclass
class SRLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class SelfRetrievalLM(LM):
    def __init__(self, retriever=None, context_window_size:int=2048, chunk_size:int=64, key_num:int=1, chunk_batch_size:int=2, add_key_continuation=False, retrieval_method="dense", order_method:str="sequential", integrate_method:str="concat", instruction:Dict=None, add_sep:Optional[List[int]]=None, debug_retrieval:bool=False, **kwds) -> None:
        super().__init__(**kwds)
        self.retriever = retriever

        assert context_window_size % chunk_size == 0, f"Make sure the context_window_size ({context_window_size}) is divisible by chunk_size ({chunk_size})!"        

        self.context_window_size = context_window_size
        self.chunk_size = chunk_size
        self.chunk_batch_size = chunk_batch_size
        self.key_num = key_num
        self.add_sep = add_sep
        self.add_key_continuation = add_key_continuation
        self.retrieval_method = retrieval_method
        self.order_method = order_method
        self.integrate_method = integrate_method
        self.debug_retrieval = debug_retrieval
        self.instruction = instruction
        
        if self.add_sep is not None:
            logger.warning(f"will add {add_sep} after retrieved chunks!")
            self.register_buffer("sep_token_ids", torch.tensor(add_sep), persistent=False)

    def _get_retrieved_chunks(self, value_chunks, retrieved_indices):
        """Get the retrieved chunks and their continuations according to retrieved_indices."""
        batch_size = value_chunks.shape[0]
        chunk_batch_size = retrieved_indices.shape[0] // batch_size

        # NOTE: by default, the retrieved_indices are sorted descendingly according to relevance
        if self.order_method == "sequential":
            retrieved_indices = retrieved_indices.sort(-1)[0]
        elif self.order_method == "relevance":
            retrieved_indices = retrieved_indices.flip(dims=(-1,))
        else:
            raise NotImplementedError(f"Order strategy {self.order_method} not implemented!")

        indices = retrieved_indices.repeat_interleave(2, -1) # batch_size * chunk_batch_size, 2k
        indices[:, 1::2] += 1

        indices = indices[..., None].expand(batch_size * chunk_batch_size, 2 * self.key_num, self.chunk_size)    # batch_size * chunk_batch_size, 2k, chunk_size
        # Slice out the retrieved chunk and its continuation from the corpus
        retrieved_chunks = value_chunks.repeat_interleave(chunk_batch_size, dim=0).gather(dim=1, index=indices).view(indices.shape[0], self.key_num, 2 * self.chunk_size)   # batch_size * chunk_batch_size, k, 2 * chunk_size
        if self.add_sep is not None:
            retrieved_chunks[..., -len(self.sep_token_ids):] = self.sep_token_ids
        retrieved_chunks = retrieved_chunks.flatten(-2, -1)
        return retrieved_chunks, retrieved_indices

    def _get_retrieved_history(self, history, retrieved_indices):
        """Get the retrieved history according to retrieved_indices."""
        batch_size = history.shape[0]

        if retrieved_indices is None:
            retrieved_history = np.array([""] * (batch_size))
        
        else:
            if isinstance(retrieved_indices, torch.Tensor):
                retrieved_indices = retrieved_indices.cpu().numpy()
            elif isinstance(retrieved_indices, np.ndarray):
                pass

            # NOTE: by default, the retrieved_indices are sorted descendingly according to relevance
            if self.order_method == "sequential":
                retrieved_indices.sort(axis=-1)
            elif self.order_method == "relevance":
                retrieved_indices = retrieved_indices[...,::-1]
            else:
                raise NotImplementedError(f"Order strategy {self.order_method} not implemented!")

            # slice out retrieved histories
            retrieved_history = np.take_along_axis(history, indices=retrieved_indices, axis=-1)
            # FIXME: I think maybe there is better way to concatenate the strings row-wise
            retrieved_history = np.array(["\n".join(x) for x in retrieved_history])
            # Last /n is important
            retrieved_history = np.char.add(retrieved_history, ["\n"] * batch_size)

        return retrieved_history

    def forward(self, **kwds):
        if "history" in kwds:
            return self.forward_with_history_retrieval(**kwds)
        else:
            return self.forward_with_chunk_retrieval(**kwds)

    def forward_with_history_retrieval(self, query:np.ndarray, history:np.ndarray, answer:np.ndarray, history_mask:torch.Tensor):
        batch_size = len(query)

        query_with_prompt = np.char.add(["Speaker 1: "] * batch_size, query)
        answer_with_prompt = np.char.add(["\nSpeaker 2: "] * batch_size, answer)
        # get answer length
        answer_length = self.tokenizer(answer.tolist(), padding=True, return_tensors="pt", return_token_type_ids=False, add_special_tokens=False)["attention_mask"].sum(-1, keepdim=True).to(self.device)

        history_size = history.shape[1]

        if self.retrieval_method == "no":
            retrieved_indices = None
        
        elif self.retrieval_method == "random":
            retrieved_indices = np.random.randint(0, history_size, (batch_size, self.key_num))
        
        elif self.retrieval_method == "recent":
            valid_history_num = history_mask.cpu().numpy().sum(axis=-1)
            valid_history_num = np.maximum(valid_history_num, self.key_num)
            start_idx = valid_history_num - self.key_num
            arange = np.arange(self.key_num)[None, :]
            retrieved_indices = arange + start_idx # batch_size, key_num

        elif self.retrieval_method == "dense":
            # masking the padded history
            history_mask = history_mask.to(self.device)

            if self.instruction is not None:
                queries = np.char.add([self.instruction["query"]] * batch_size, query)
                keys = np.char.add([self.instruction["key"]] * batch_size, history.reshape(-1))
            else:
                queries = query
                keys = history.reshape(-1)
            history_embedding = self.retriever.encode(keys.tolist()).unflatten(0, (batch_size, history_size))  # B * N, D
            context_embedding = self.retriever.encode(queries.tolist())    # B, D
            scores = torch.einsum("bnd,bd->bn", history_embedding, context_embedding)   # B, N
            # mask padded histories
            scores = scores.masked_fill(~history_mask, torch.finfo(scores.dtype).min)
            _, retrieved_indices = scores.topk(k=self.key_num, dim=-1) # B, K
        
        elif self.retrieval_method == "bm25":
            retrieved_indices = np.zeros(batch_size, self.key_num, dtype=np.int32)

            for batch_idx in range(batch_size):
                bm25 = deepcopy(self.retriever)
                bm25.index(history[batch_idx].tolist())
                _, indice = bm25.search(query[batch_idx].tolist(), hits=self.key_num)
                retrieved_indices[batch_idx] = indice[0]

        elif self.retrieval_method == "oracle":
            assert self.key_num == 1 and batch_size == 1, f"Retrieval_method 'oracle' is only available when k == 1 and batch_size == 1!"

            min_loss = 1e3
            min_k = 0
            min_outputs = None
            
            for hist_idx in range(history_size):
                hist = history[:, hist_idx]
                inputs = np.char.add(hist, ["\n"] * batch_size)
                inputs = np.char.add(inputs, query_with_prompt)
                inputs = np.char.add(inputs, answer_with_prompt)

                inputs = self.tokenizer(inputs.tolist(), padding=True, truncation=True, max_length=self.context_window_size, return_tensors="pt", return_token_type_ids=False).to(self.device)

                labels = inputs["input_ids"].clone()
                arange = torch.arange(labels.shape[1] - 1, -1, -1, device=self.device).expand(labels.shape)
                labels_mask = arange >= answer_length
                inputs["labels"] = labels.masked_fill(labels_mask, -100)
                outputs = self.model(**inputs)
                loss = outputs.loss

                # print(self.tokenizer.batch_decode(labels.masked_fill(labels_mask, self.tokenizer.pad_token_id)))
                # print(inputs["input_ids"])
                # print(inputs["labels"])
                # save_pickle(inputs.to("cpu"), "debug.pkl")
                # print(loss)
                # input()

                if loss < min_loss:
                    min_loss = loss
                    min_k = hist_idx
                    min_outputs = outputs

            if self.debug_retrieval:
                print(min_k)
                print(f"***Query***\n{query[0].tolist()}")
                print(f"***Answer***\n{answer[0].tolist()}")
                print(f"***Retrieved***\n{history[0, min_k].tolist()}")
                print(outputs.loss)
                input()
            return min_outputs

        else:
            raise NotImplementedError(f"Retrieval method {self.retrieval_method} not implemented!")

        retrieved_history = self._get_retrieved_history(history, retrieved_indices)

        # combine retrieved turns with the current context
        inputs = np.char.add(retrieved_history, query_with_prompt)
        inputs = np.char.add(inputs, answer_with_prompt)

        inputs = self.tokenizer(inputs.tolist(), padding=True, truncation=True, max_length=self.context_window_size, return_tensors="pt", return_token_type_ids=False).to(self.device)

        labels = inputs["input_ids"].clone()
        arange = torch.arange(labels.shape[1] - 1, -1, -1, device=self.device).expand(labels.shape)
        labels_mask = arange >= answer_length
        inputs["labels"] = labels.masked_fill(labels_mask, -100)

        # print(self.tokenizer.batch_decode(labels.masked_fill(labels_mask, self.tokenizer.pad_token_id)))

        outputs = self.model(**inputs)
        if self.debug_retrieval:
            for i in range(batch_size):
                print(f"***Query***\n{query[i].tolist()}")
                print(f"***Answer***\n{answer[i].tolist()}")
                print(f"***Retrieved***\n{retrieved_history[i].tolist()}")
                print(outputs.loss)
            input()
        return outputs
    
    def forward_with_chunk_retrieval(self, input_ids, attention_mask, labels):
        batch_size, inputs_length = input_ids.shape

        # in this case, all inputs are visible to the language model, thus no retrieval needed
        if self.retrieval_method == "no":
            input_ids = input_ids[:, -self.context_window_size:]
            attention_mask = attention_mask[:, -self.context_window_size:]
            labels = labels[:, -self.context_window_size:]
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs

        # Pad inputs to multiple of chunk_size
        num_chunks = math.ceil(inputs_length / self.chunk_size)
        # NOTE: get the minor one because some inputs may be shorter than context_window_size even after padding to multiple of chunk size
        context_window_size = min(num_chunks * self.chunk_size, self.context_window_size)
        if inputs_length % self.chunk_size != 0:
            pad_length = num_chunks * self.chunk_size - inputs_length
            input_ids = torch.cat([input_ids.new_zeros(batch_size, pad_length) + self.tokenizer.pad_token_id, input_ids], dim=-1)
            attention_mask = torch.cat([attention_mask.new_zeros(batch_size, pad_length), attention_mask], dim=-1)
            labels = torch.cat([labels.new_zeros(batch_size, pad_length) - 100, labels], dim=-1)
            inputs_length = input_ids.shape[1]

        # Find the start of target. All retrieval operation starts from the preceeding chunk to the target
        is_valid = (labels != -100).float()
        target_start_index = is_valid.argmax(-1)
        assert (target_start_index == target_start_index[0]).all(), f"Make sure all targets in the batch starts from the same token index!"
        target_start_index = target_start_index[0].item()
        assert target_start_index % self.chunk_size == 0, f"Make sure the target_length ({inputs_length} - {target_start_index} = {inputs_length - target_start_index}) is divisible by chunk_size ({self.chunk_size})!"

        # Organize inputs
        n_target_chunk = (inputs_length - target_start_index) // self.chunk_size
        n_window_chunk = context_window_size // self.chunk_size
        input_ids = input_ids.view(batch_size, -1, self.chunk_size)
        labels = labels[:, -context_window_size:]
        # print(labels)

        # Split queries, keys and values
        # the chunk preceeding target is the first query
        query_chunks = input_ids[:, -n_target_chunk - 1: -1]
        if self.integrate_method == "replace":
            assert n_window_chunk >= (n_target_chunk + 1 + 2 * self.key_num), f"Make sure there are at least k * 2 + 1 + n_target_chunk = {self.key_num * 2 + 1 + n_target_chunk} chunks (found {context_window_size} / {self.chunk_size} = {n_window_chunk}) that can be replaced with retrieved contents!"
            # these tokens will be directly concatenated with retrieved chunks
            fixed_context = input_ids[:, -n_window_chunk + 2 * self.key_num:]
            # besides previous chunks, the last chunk is also taken as keys because
            # we only want to replace the context when there are more relevant ones
            key_chunks = input_ids[:, :-n_window_chunk + 1]
            if self.add_key_continuation:
                continuation_chunks = input_ids[:, 1: -n_window_chunk + 2]
                key_chunks = torch.cat([key_chunks, continuation_chunks], dim=-1)
            # value chunks extend key chunks by one chunk because we may want to splice out the continuation chunk of the last key
            value_chunks = input_ids[:, :-n_window_chunk + 2]
            labels_mask_indices_offset = 0
        elif self.integrate_method == "concat":
            fixed_context = input_ids[:, -n_window_chunk:]
            key_chunks = input_ids[:, :-n_window_chunk - 1]
            if self.add_key_continuation:
                continuation_chunks = input_ids[:, 1: -n_window_chunk]
                key_chunks = torch.cat([key_chunks, continuation_chunks], dim=-1)
            value_chunks = input_ids[:, :-n_window_chunk]
            labels = torch.cat([labels.new_zeros(batch_size, 2 * self.key_num * self.chunk_size) - 100, labels], dim=-1)
            labels_mask_indices_offset = 2 * self.key_num * self.chunk_size
        else:
            raise NotImplementedError(f"Integration strategy {self.integrate_method} not implemented!")
        fixed_context = fixed_context.flatten(-2, -1)

        # Prepare labels mask to be used in sub-batch
        # Each query chunk will produce a sample, but only its next chunk should be evaluated
        n_query_chunk = query_chunks.shape[1]
        n_key_chunk = key_chunks.shape[1]
        target_chunk_start_idx = n_window_chunk - n_target_chunk
        # How many tokens in total until i-th chunk
        bias = torch.arange(n_query_chunk, device=input_ids.device) * self.chunk_size
        # Inside each chunk, the indices start from 0 to chunk_size - 1
        # add target_chunk_start_idx because we want the labels computed 
        # only for target chunks
        arange = torch.arange(self.chunk_size, device=input_ids.device) + target_chunk_start_idx * self.chunk_size
        labels_mask_indices = bias[:, None] + arange[None, :]
        labels_mask_indices = labels_mask_indices.view(n_query_chunk, self.chunk_size) + labels_mask_indices_offset

        if self.retrieval_method == "dense":
            # Encode queries and keys
            queries = self.tokenizer.batch_decode(query_chunks.flatten(0, 1), skip_special_tokens=True)
            keys = self.tokenizer.batch_decode(key_chunks.flatten(0, 1), skip_special_tokens=True)
            if self.instruction is not None:
                queries = [self.instruction["query"] + q for q in queries]
                keys = [self.instruction["key"] + k for k in keys]
            # The retriever automatically does truncation and padding
            query_embeddings = self.retriever.encode(queries).view(batch_size, n_query_chunk, -1)
            key_embeddings = self.retriever.encode(keys).view(batch_size, n_key_chunk, -1)

        elif self.retrieval_method == "random":
            pass
        
        elif self.retrieval_method == "bm25":
            bm25_indexes = []
            for i in range(batch_size):
                bm25 = deepcopy(self.retriever)
                bm25.index(key_chunks[i].tolist())
                bm25_indexes.append(bm25)

        elif self.retrieval_method == "oracle":
            assert self.key_num == 1 and batch_size == 1, f"Retrieval_method 'oracle' is only available when k == 1 and batch_size == 1!"
            all_losses = 0
            all_valid_tokens = 0
            # enumerate all chunks
            for i in range(n_query_chunk):
                min_k = 0
                min_loss = 1e3
                min_retrieved_chunks = None
                min_input_ids = None

                sub_labels = labels    # batch_size, n_window_chunk * self.chunk_size
                sub_labels_mask = torch.ones_like(sub_labels, dtype=torch.bool)
                sub_labels_mask.scatter_(dim=-1, index=labels_mask_indices[None, i].expand(batch_size, -1), value=False)
                sub_labels = sub_labels.masked_fill(sub_labels_mask, -100)
                # NOTE: the loss is averaged over valid tokens, thus we must store the valid token number for the final computation
                valid_tokens = (sub_labels != -100).sum()

                for k in range(n_key_chunk):
                    retrieved_chunks = value_chunks[:, k: k+2]  # batch_size, 2, chunk_size
                    retrieved_chunks = retrieved_chunks.flatten(-2, -1)
                    if self.add_sep is not None:
                        retrieved_chunks[..., -len(self.sep_token_ids):] = self.sep_token_ids

                    sub_input_ids = torch.cat([retrieved_chunks, fixed_context], dim=-1)
                    sub_attention_mask = (sub_input_ids != self.tokenizer.pad_token_id).long()

                    outputs = self.model(input_ids=sub_input_ids, attention_mask=sub_attention_mask, labels=sub_labels)
                    if (sub_labels == -100).all():
                        # NOTE: in this case, the model will return nan. We correct its behavior by returning 0
                        loss = 0
                    else:
                        loss = outputs.loss
                    
                    if loss < min_loss:
                        min_loss = loss
                        min_k = k
                        min_retrieved_chunks = retrieved_chunks
                        min_input_ids = sub_input_ids

                if self.debug_retrieval:
                    print("-"*50)
                    context = fixed_context.unflatten(-1, (-1, self.chunk_size))
                    print(min_loss)
                    print(f"***Indices***\n{min_k}")
                    print(f"***Query***\n{repr(self.tokenizer.decode(query_chunks[0, i]))}")
                    print(f"***Target***\n{repr(self.tokenizer.decode(context[0, -n_target_chunk]))}")
                    print(f"***Retrieved***\n{repr(self.tokenizer.decode(min_retrieved_chunks[0]))}")
                    print(f"***Inputs***\n{repr(self.tokenizer.decode(min_input_ids[0]))}")
                    print(f"***Labels***\n{repr(self.tokenizer.decode(sub_labels.masked_fill(sub_labels_mask, self.tokenizer.pad_token_id)[0]))}")
                    print()
                    input()

                all_losses += min_loss * valid_tokens
                all_valid_tokens += valid_tokens
        
            loss = all_losses / all_valid_tokens
            return SRLMOutput(loss=loss)
        else:
            raise NotImplementedError(f"Retrieval method {self.retrieval_method} not implemented!")

        # Compute language modeling loss for each target chunk in sub-batch
        all_losses = None
        all_valid_tokens = 0
        for i in range(0, n_query_chunk, self.chunk_batch_size):
            j = min(i + self.chunk_batch_size, n_query_chunk)
            chunk_batch_size = j - i

            if self.retrieval_method == "dense":
                query_embedding = query_embeddings[:, i: j]   # batch_size, chunk_batch_size, d_embed
                rel_score = torch.einsum("bid,bjd->bij", query_embedding, key_embeddings) # batch_size, chunk_batch_size, n_key_chunk
                retrieved_indices = rel_score.topk(self.key_num, dim=-1)[1].flatten(0, 1)    # batch_size * chunk_batch_size, k

            elif self.retrieval_method == "random":
                retrieved_indices = torch.randint(0, n_key_chunk, (batch_size * chunk_batch_size, self.key_num), device=input_ids.device)
                
            elif self.retrieval_method == "bm25":
                retrieved_indices = torch.zeros(batch_size, chunk_batch_size, self.key_num, dtype=torch.long, device=value_chunks.device)
                for batch_idx in range(batch_size):
                    query_chunk = query_chunks[batch_idx, i: j].tolist()
                    _, indice = bm25_indexes[batch_idx].search(query_chunk, hits=self.key_num)
                    retrieved_indices[batch_idx] = torch.from_numpy(indice)
                retrieved_indices = retrieved_indices.flatten(0, 1)

            # batch_size * chunk_batch_size, k * 2 * chunk_size
            retrieved_chunks, retrieved_indices = self._get_retrieved_chunks(value_chunks, retrieved_indices)
            
            # Each sub-batch has its own retrieved contexts
            sub_input_ids = torch.cat([retrieved_chunks, fixed_context.repeat_interleave(chunk_batch_size, dim=0)], dim=-1)
            sub_attention_mask = (sub_input_ids != self.tokenizer.pad_token_id).long()

            # NOTE: here we donot add position_ids to keep the outputs exactly the same as the default behavior
            # position_ids = attention_mask.cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask == 0, 0)

            # repeat labels across sub-batch
            sub_labels = labels.repeat_interleave(chunk_batch_size, dim=0)    # batch_size * chunk_batch_size, n_window_chunk * self.chunk_size
            sub_labels_mask = torch.ones_like(sub_labels, dtype=torch.bool)
            # NOTE: only compute loss for this sub-batch
            sub_labels_mask.scatter_(dim=-1, index=labels_mask_indices[None, i: j].expand(batch_size, -1, -1).flatten(0, 1), value=False)
            sub_labels = sub_labels.masked_fill(sub_labels_mask, -100)

            if self.debug_retrieval:
                print("-"*50)
                context = fixed_context.unflatten(-1, (-1, self.chunk_size))
                indices = retrieved_indices.unflatten(0, (batch_size, chunk_batch_size))
                chunks = retrieved_chunks.view(batch_size, chunk_batch_size, self.key_num, 2 * self.chunk_size)
                for r in range(chunk_batch_size):
                    idx = r + i
                    print(f"***Indices***\n{indices[0, r]}")
                    print(f"***Query***\n{repr(self.tokenizer.decode(query_chunks[0, idx]))}")
                    print(f"***Target***\n{repr(self.tokenizer.decode(context[0, -n_target_chunk + idx]))}")
                    print(f"***Retrieved***\n{repr(self.tokenizer.batch_decode(chunks[0, r]))}")
                    print(f"***Inputs***\n{repr(self.tokenizer.batch_decode(sub_input_ids))}")
                    print(f"***Labels***\n{repr(self.tokenizer.batch_decode(sub_labels.masked_fill(sub_labels_mask, self.tokenizer.pad_token_id)))}")
                    print()
                input()

            outputs = self.model(input_ids=sub_input_ids, attention_mask=sub_attention_mask, labels=sub_labels)
            if (sub_labels == -100).all():
                # NOTE: in this case, the model will return nan. We correct its behavior by returning 0
                loss = 0
            else:
                loss = outputs.loss
            # NOTE: the loss is averaged over valid tokens, thus we must store the valid token number for the final computation
            valid_tokens = (sub_labels != -100).sum()

            if all_losses is None:
                all_losses = loss * valid_tokens
            else:
                all_losses += loss * valid_tokens
            all_valid_tokens += valid_tokens
        
        loss = all_losses / all_valid_tokens
        return SRLMOutput(loss=loss)

    @torch.no_grad()
    def compute_perplexity(self, dataloader):
        """
        Compute perplexity over long inputs
        """
        self.model.eval()
        all_nlls = []
        for step, inputs in enumerate(tqdm(dataloader, desc='Computing Perplexity')):
            # if step > 5:
            #     break
            # move to gpu
            inputs = self._move_to_device(inputs)
            outputs = self(**inputs)
            nll = outputs.loss

            if self.accelerator is not None:
                # mean nlls from all processes
                nll = self.accelerator.gather_for_metrics(nll).mean()

            all_nlls.append(nll.tolist())

        all_nlls = sum(all_nlls) / len(all_nlls)
        perplexity = math.exp(all_nlls)
        return perplexity

    # TODO
    # def generate(self, input_ids, attention_mask, **kwds):
    #     """Generate by chunks"""
    #     generation_config = self.model.generation_config
    #     assert generation_config.max_new_tokens is not None, f"Make sure the max_new_tokens parameter in model's generation_config is not None!"
    #     global_max_new_tokens = generation_config.max_new_tokens
    #     n_generate_chunk = global_max_new_tokens // self.chunk_size
    #     batch_size = input_ids.shape[0]

    #     assert input_ids.shape[1] % self.chunk_size == 0, f"Make sure the generation input length {input_ids.shape[1]} is divisible by chunk size!"

    #     # 1. Encode
    #     n_window_chunk = input_ids.shape[1] // self.chunk_size
    #     # concatenate extra context
    #     if prev_input_ids is not None:
    #         assert prev_input_ids.shape[1] % self.chunk_size == 0, f"Make sure the prev input length {prev_input_ids.shape[1]} is divisible by chunk size!"
    #         input_ids = torch.cat([prev_input_ids, input_ids], dim=-1)
    #     input_ids = input_ids.view(batch_size, -1, self.chunk_size)

    #     key_chunks = input_ids[:, :-n_window_chunk + 2 * self.key_num - 1]
    #     value_chunks = input_ids[:, :-n_window_chunk + 2 * self.key_num]
    #     fixed_context = input_ids[:, -n_window_chunk + 2 * self.key_num:]  # batch_size, n_window_chunk - 2 * k, chunk_size

    #     n_key_chunk = key_chunks.shape[1]
    #     keys = self.tokenizer.batch_decode(key_chunks.flatten(0, 1), skip_special_tokens=True)
    #     key_embeddings = self.encoder(keys).view(batch_size, n_key_chunk, -1)

    #     # 2. Generate by chunk
    #     for step in range(n_generate_chunk):
    #         query_chunk = fixed_context[:, -1:]  # batch_size, 1, chunk_size
    #         query = self.tokenizer.batch_decode(query_chunk.squeeze(1), skip_special_tokens=True)
    #         query_embedding = self.encoder(query).view(batch_size, 1, -1)
    #         # Slice out the retrieved chunk and its continuation from the corpus
    #         retrieved_chunks, retrieved_indices = self._dense_retrieval(query_embedding, key_embeddings, value_chunks)
    #         if self.debug_retrieval:
    #             print("-"*50)
    #             indices = retrieved_indices.unflatten(0, (batch_size, 1))
    #             chunks = retrieved_chunks.view(batch_size, 1, self.key_num, 2 * self.chunk_size)
    #             print(f"***Indices***\n{indices[0, 0]}")
    #             print(f"***Query***\n{repr(self.tokenizer.decode(query_chunk[0, 0]))}")
    #             print(f"***Retrieved***\n{repr(self.tokenizer.batch_decode(chunks[0, 0]))}")
    #             print()
    #             input()

    #         step_input_ids = torch.cat([retrieved_chunks, fixed_context.flatten(-2, -1)], dim=-1)
    #         step_attention_mask = (step_input_ids != self.tokenizer.pad_token_id).long()
    #         # generate chunk_size tokens once
    #         kwds["max_new_tokens"] = self.chunk_size
    #         outputs = self.model.generate(input_ids=step_input_ids, attention_mask=step_attention_mask, **kwds) # batch_size, chunk_size
    #         # slice out the newly-generated tokens
    #         outputs = outputs[:, step_input_ids.shape[1]:]   # batch_size, chunk_size
    #         assert outputs.shape[-1] == self.chunk_size

    #         fixed_context = torch.cat([fixed_context, outputs.unsqueeze(1)], dim=1) # batch_size, -, chunk_size

    #     # 3. Finalize. Set all tokens after the first eos token to pad token
    #     generated_tokens = torch.cat([input_ids[:, -n_window_chunk: -n_window_chunk + 2 * self.key_num:], fixed_context], dim=1).flatten(-2, -1)    # batch_size, (n_window_chunk + n_generate_chunk) * chunk_size
    #     # is_eos = (generated_tokens == self.tokenizer.eos_token_id).float()
    #     # has_eos = (generated_tokens == self.tokenizer.eos_token_id).any(-1)
    #     # eos_start_index = is_eos.argmax(-1)
    #     # print(generated_tokens)
    #     # print(eos_start_index, has_eos)
    #     # for i, idx in enumerate(eos_start_index):
    #     #     if has_eos[i]:
    #     #         generated_tokens[i, idx + 1:] = self.tokenizer.pad_token_id

    #     return generated_tokens
