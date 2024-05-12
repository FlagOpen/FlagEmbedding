import torch
import numpy as np
import torch.distributed as dist
from transformers.utils import logging
from typing import List, Tuple, Optional
from .modeling_retrieval import BM25Retriever

logger = logging.get_logger(__name__)


class Memory(torch.nn.Module):
    def __init__(self, model_config, beacon_window:int=1024, beacon_stride:List[int]=[512], beacon_attn:str="step-expansion", beacon_attend_previous:bool=True, beacon_ratio:List[int]=[8], beacon_stride_mix:str="step-random", beacon_ratio_mix:str="step-random", beacon_param:List[str]=["q", "k", "v", "o"], k_seq_dim:int=2, v_seq_dim:int=2, retrieval_method:str=None, retrieval_topk:int=2) -> None:
        super().__init__()

        for stride in beacon_stride:
            assert beacon_window >= stride, f"Make sure the beacon_window {beacon_window} >= beacon_stride {stride}!"
        assert beacon_attn in ["segmentation", "step-expansion", "full-coverage"], f"beacon_attn {beacon_attn} not implemented!"
        assert beacon_stride_mix in ["instance-random", "step-random", "mix-random"], f"beacon_stride_mix {beacon_stride_mix} not implemented!"
        assert beacon_ratio_mix in ["instance-random", "step-random", "mix-random", "sequence"] or "adapt-" in beacon_ratio_mix, f"beacon_ratio_mix {beacon_ratio_mix} not implemented!"

        if retrieval_method == "bm25":
            assert len(beacon_stride) == 1, f"Currently retrieval do not support dynamic strides."
            assert retrieval_topk >= 2, f"Make sure retrieval_topk >= 2. Found {retrieval_topk}."
            assert len(beacon_ratio) == 2, f"Make sure there are two beacon ratios specified, one for retrieved windows and the other for non-retrieved windows. Found {self.beacon_ratio}"

        info = f"applying activation beacon on {beacon_param}, with window size {beacon_window}, stride {beacon_stride} (mixed by {beacon_stride_mix}), {beacon_attn} attention ({'attending to previous beacons' if beacon_attend_previous else 'not attending to previous beacons'}), condensing ratio {beacon_ratio} (mixed by {beacon_ratio_mix}), {retrieval_method+' retrieval'+' top-'+str(retrieval_topk) if retrieval_method is not None else 'no retrieval'}, ..."
        logger.info(info)

        self.beacon_window = beacon_window
        self.beacon_stride = beacon_stride
        self.beacon_attn = beacon_attn
        self.beacon_ratio = beacon_ratio
        self.beacon_stride_mix = beacon_stride_mix
        self.beacon_ratio_mix = beacon_ratio_mix
        max_beacon_size = max([beacon_window // x for x in beacon_ratio if x > 0] + [1])
        self.beacon_tokens = torch.zeros(max_beacon_size, dtype=torch.long) + model_config.vocab_size
        
        # initialize necessary parameters
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_layers = model_config.num_hidden_layers
        self.max_position_embeddings = model_config.max_position_embeddings

        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk

        self.rng = np.random.default_rng(42)
        self.reset()
    
    @property
    def finish(self):
        return self.end_idx == self.sequence_length
    
    def get_memory_size(self):
        beacon_memory_size = 0
        raw_memory_size = 0
        if self.beacon_activations[0][0] is not None:
            beacon_memory_size += self.beacon_activations[0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        memory_size = beacon_memory_size + raw_memory_size
        return beacon_memory_size, raw_memory_size, memory_size

    def reset(self):
        # the length of current sequence
        self.sequence_length = 0
        # the length of all sequences until the memory is reset
        self.total_sequence_length = 0
        # the cursor pointing to the start of the current window
        self.start_idx = 0
        # the cursor pointing to the end of the current window
        self.end_idx = 0
        # the beacon sizes of all strides
        self._beacon_sizes = []
        # the step index
        self.step_idx = 0

        if self.beacon_ratio_mix != "step-random":
            self._stride = None
            self._ratio = None

        self.batch_loss = None
        self.valid_token_num = None

        self.raw_activations = [(None, None) for _ in range(self.num_layers)]
        self.beacon_activations = [(None, None) for _ in range(self.num_layers)]

        if self.retrieval_method == "bm25":
            self.retriever = BM25Retriever()
            self.beacon_ratio_mix = "retrieval"

        # NOTE: when training, we strictly aligh the rng_state across processes
        if self.training and dist.is_initialized():
            rng_state = self.rng.__getstate__()
            if dist.get_rank() == 0:
                obj = [rng_state]
            else:
                obj = [None]
            dist.broadcast_object_list(obj, src=0)
            self.rng.__setstate__(obj[0])

    def prepare(self, input_ids, attention_mask, labels):
        """
        Prepare inputs for the model.
        """
        # TODO: support batch_size > 1?
        assert input_ids.shape[0] == 1, f"Make sure batch_size is 1!"
        
        # NOTE: rebase the start/end idx so that it becomes an offset from the start of the current sequence
        self.start_idx -= self.sequence_length
        self.end_idx -= self.sequence_length

        self.sequence_length = input_ids.shape[1]
        self.total_sequence_length += input_ids.shape[1]

        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:], labels.new_zeros((labels.shape[0], 1)) - 100], dim=-1)

        # if the current sequence has been completely processed
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

        # TODO: retrieval on specified steps
        # TODO: retrieval for future inputs
        # TODO: different retrieval methods
        if self.retrieval_method == "bm25" and self.step_idx == 0:
            index = BM25Retriever()
            window = self.beacon_window
            stride = self.beacon_stride[0]

            input_ids = input_ids[0].tolist()
            corpus = [input_ids[:window]]
            for j in range(window, len(input_ids), stride):
                corpus.append(input_ids[j: j + stride])

            # NOTE: use the last 32 token as query (very naive heuristics)
            query = input_ids[-32:]
            index.index(corpus)
            topk_scores, topk_indices = index.search(query, hits=self.retrieval_topk)
            topk_indices = set([x for x in topk_indices[0] if x > -1])

            self._topk_indices = topk_indices

    def set_stride(self):
        """Choose a stride from self.beacon_stride"""
        beacon_stride = self.beacon_stride
        
        if len(beacon_stride) == 1:
            return beacon_stride[0]

        if self.beacon_stride_mix == "mix-random":
            stride_mix = self.rng.choice(["instance-random", "step-random"]).tolist()
        else:
            stride_mix = self.beacon_stride_mix
        
        if stride_mix == "instance-random":
            if self._beacon_stride is None:
                stride = self.rng.choice(beacon_stride).tolist()
                self._stride = stride
            else:
                stride = self._stride

        elif stride_mix == "step-random":
            stride = self.rng.choice(beacon_stride).tolist()

        else:
            raise NotImplementedError

        return stride

    def set_condensing_ratio(self, beacon_stride, start_idx, end_idx):
        """Choose a condensing ratio from self.beacon_ratio"""
        def filter_ratio(ratios, stride):
            valid_ratios = []
            for ratio in ratios:
                # stride must be bigger than condensing ratio because we there must be at least one beacon
                if stride < ratio:
                    continue
                # step-expansion and segmentation requires the stride to be evenly divisible by condensing ratio
                if self.beacon_attn != "full-coverage" and ratio > 0 and (stride % ratio) != 0:
                    continue
                # when training, ratio=0 is valid if previous windows contain beacon or later windows contain beacon
                if ratio == 0 and self.training:
                    previous_beacons = [b for b in self._beacon_sizes if b != -1]
                    following_beacons = (start_idx + stride + self.beacon_window) <= self.sequence_length
                    if len(previous_beacons) == 0 and not following_beacons:
                        continue
                valid_ratios.append(ratio)
            assert len(valid_ratios), f"Cannot find valid condensing ratio (among {ratios}) for stride {stride}!"
            return valid_ratios

        def get_max_length(ratios):
            max_lengths = []
            for condensing_ratio in ratios:
                if condensing_ratio > 0:
                    max_lengths.append((self.max_position_embeddings - self.beacon_window) * condensing_ratio + self.beacon_window)
                else:
                    max_lengths.append(self.max_position_embeddings)
            return max_lengths

        if len(self.beacon_ratio) == 1:
            return self.beacon_ratio[0]

        beacon_ratio = filter_ratio(self.beacon_ratio, beacon_stride)

        if self.beacon_ratio_mix == "mix-random":
            ratio_mix = self.rng.choice(["instance-random", "step-random"]).tolist()
        else:
            ratio_mix = self.beacon_ratio_mix

        if ratio_mix == "instance-random":
            if self._ratio is None:
                beacon_ratio = self.rng.choice(beacon_ratio).tolist()
                self._ratio = beacon_ratio
            else:
                beacon_ratio = self._ratio

        elif ratio_mix == "step-random":
            beacon_ratio = self.rng.choice(beacon_ratio).tolist()
        
        elif ratio_mix == "sequence":
            idx = min(self.step_idx, len(beacon_ratio) - 1)
            beacon_ratio = beacon_ratio[idx]
        
        elif ratio_mix == "retrieval":
            # for retrieved windows, we use low ratio; otherwise high ratio
            if self.step_idx in self._topk_indices:
                beacon_ratio = min(self.beacon_ratio)
            else:
                beacon_ratio = max(self.beacon_ratio)

        elif "adapt" in ratio_mix:
            if self._ratio is None:
                future_length = int(ratio_mix.split("-")[1])
                sequence_length = self.total_sequence_length + future_length
                max_lengths = get_max_length(beacon_ratio)
                # ascendingly sort the max lengths
                valid_max_lengths_and_indices = [x for x in enumerate(max_lengths) if x[1] >= sequence_length]
                if len(valid_max_lengths_and_indices):
                    minimum_length_index = min(valid_max_lengths_and_indices, key=lambda x: x[1])[0]
                    # use the minimal possible length for this sequence (the smallest fold ratio)
                    beacon_ratio = beacon_ratio[minimum_length_index]
                else:
                    beacon_ratio = max(beacon_ratio)
                    # logger.warning(f"Failed to find valid fold window and size for sequence length {sequence_length}, as the maximum theoretical length is {max(max_lengths)}. Fall back to use the maximum one: {beacon_ratio}.")
                self._ratio = beacon_ratio
            else:
                beacon_ratio = self._ratio

        return beacon_ratio

    def step(self):
        """
        Yield one window with the following logic:

        The window size is L, the stride is S.
        The window moves over S tokens at a time. The raw activations passed by the window are condensed according to a condensing_ratio.
        The beacons are added if and only if the raw activations fulfill the window.
        In the future, we may switch window size to decrease cache size of raw activations.
        """
        # the starting position of the current window w.r.t. the start of the current input sequence
        start_idx = self.start_idx
        # the end position of the current window w.r.t. the start of the current input sequence
        end_idx = start_idx + self.beacon_window

        # indicates if the current window is completely filled by raw activations and new tokens
        # we only append beacon tokens for full windows
        is_full_window = True
        if end_idx > self.sequence_length:
            # the input is shorter than the initial window size
            end_idx = self.sequence_length
            is_full_window = False

        # the real window size (remaining_size + new_token_size)
        window_size = end_idx - start_idx

        if is_full_window:
            # set stride and condensing ratio
            beacon_stride = self.set_stride()
            condensing_ratio = self.set_condensing_ratio(beacon_stride, start_idx=start_idx, end_idx=end_idx)

            # the stride must be evenly divisible by condensing_ratio
            if condensing_ratio > 0:
                beacon_size = beacon_stride // condensing_ratio
            else:
                # the raw activations are used as beacon activations
                beacon_size = -1
            # forward start_idx and end_idx
            next_start_idx = start_idx + beacon_stride
            # how many raw activations to save
            raw_size_to_cache = end_idx - next_start_idx
            self.remaining_size = 0
        else:
            # no stride because the sequence has finished
            next_start_idx = start_idx
            # cache all recent raw activations to be used in the next window
            raw_size_to_cache = window_size
            self.remaining_size = window_size
            beacon_size = 0

        # this is for debugging the resilience to different beacons
        # if self.step_idx == 97:
        #     a = torch.load("beacon_activations")
        #     for i, (beacon_key, beacon_value) in enumerate(self.beacon_activations):
        #         foreign_beacon_key = a[i][0]
        #         foreign_beacon_value = a[i][1]
        #         new_beacon_key = cat_tensor([foreign_beacon_key, slice_tensor(beacon_key, start=16, dim=self.k_seq_dim)], dim=self.k_seq_dim)
        #         new_beacon_value = cat_tensor([foreign_beacon_value, slice_tensor(beacon_value, start=16, dim=self.v_seq_dim)], dim=self.v_seq_dim)
        #         new_beacon_key = foreign_beacon_key
        #         new_beacon_value = foreign_beacon_value
        #         self.beacon_activations[i] = (new_beacon_key, new_beacon_value)

        # streamingly add new input_ids
        input_ids = self.input_ids[:, self.end_idx: end_idx]
        batch_size = input_ids.shape[0]
        if self.attention_mask is not None:
            attention_mask = self.attention_mask[:, self.end_idx: end_idx]
        else:
            attention_mask = torch.ones_like(input_ids)
        if self.labels is not None:
            labels = self.labels[:, self.end_idx: end_idx]
        else:
            labels = None
        # prepend 1 to attention mask for previous memory
        _, _, memory_size = self.get_memory_size()
        if memory_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, memory_size), attention_mask], dim=1)
        
        # append beacons if necessary
        if is_full_window and beacon_size > 0:
            input_ids = torch.cat([input_ids, self.beacon_tokens[:beacon_size].expand(batch_size, -1).to(input_ids.device, dtype=input_ids.dtype)], dim=1)
            # NOTE: prepend beacon_memory_size 1 to attention_mask because we have past_key_values
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, beacon_size)], dim=1)
            if labels is not None:
                labels = torch.cat([labels, labels.new_zeros(batch_size, beacon_size) - 100], dim=1)

        # generate memory (memory_length = old_beacon_size + beacon_size * condensing_ratio + raw_cache_size)
        past_key_values = []
        for (beacon_key, beacon_value), (raw_key, raw_value) in zip(self.beacon_activations, self.raw_activations):
            key = cat_tensor([beacon_key, raw_key], dim=self.k_seq_dim)
            value = cat_tensor([beacon_value, raw_value], dim=self.v_seq_dim)
            layer_past_key_values = (key, value, beacon_size, raw_size_to_cache, window_size)
            past_key_values.append(layer_past_key_values)

        # involked in self.output()
        self._beacon_sizes.append(beacon_size)
        # update end_idx
        self.start_idx = next_start_idx
        self.end_idx = end_idx
        self.step_idx += 1

        # print("****************************************")
        # if is_full_window:
        #     print(f"total_seq_len:      {self.total_sequence_length}")
        #     print(f"stride:             {beacon_stride}")
        #     print(f"condensing ratio:   {condensing_ratio}")
        #     print(f"beacon_size:        {beacon_size}")
        # print(f"input_ids:          {input_ids.shape}")
        # print(f"start_idx:          {start_idx}")
        # print(f"next_start_idx:     {next_start_idx}")
        # print(f"end_idx:            {end_idx}")
        # x = input()
        # if x == "s":
        #     return
        # if self.step_idx == 3:
        #     input()

        return input_ids, attention_mask, past_key_values, labels

    def update_memory(self, past_key_values):
        """
        Accumulate beacon activations and raw activations.
        """
        for layer_idx, (key, value, beacon_size, raw_size_to_cache, window_size) in enumerate(past_key_values):
            # NOTE: the past_key_values are incrementally returned (only the new keys and values are returned)

            # key/value: (num_layer, 2, batch_size, num_head, new_seq_len, head_dim)
            # beacon_size: how many beacon activations are in key and value
            # raw_size_to_cache: how many raw activations should be kept

            previous_beacon_key, previous_beacon_value = self.beacon_activations[layer_idx]
            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]
            
            if beacon_size == 0:
                # this means the current input does not fulfill a window
                # thus, the key and value are all raw activations, and we accumulate them until the window is fulfilled
                beacon_key = previous_beacon_key
                beacon_value = previous_beacon_value

                assert raw_size_to_cache == window_size
                raw_key = cat_tensor([
                    previous_raw_key,
                    key
                ], dim=self.k_seq_dim)
                raw_value = cat_tensor([
                    previous_raw_value, 
                    value
                ], dim=self.v_seq_dim)

            elif beacon_size == -1:
                # this means the raw activations are used as beacon activations for this window

                if raw_size_to_cache > 0:
                    # if we have raw activations, we must first concatenate previous raw activations and current ones, then extract raw_size_to_cache as raw memory, while others as beacon memory
                    concat_key = cat_tensor([
                        previous_raw_key, 
                        key
                    ], dim=self.k_seq_dim)
                    concat_value = cat_tensor([
                        previous_raw_value, 
                        value
                    ], dim=self.v_seq_dim)

                    beacon_key = cat_tensor([
                        previous_beacon_key,
                        slice_tensor(concat_key, end=-raw_size_to_cache, dim=self.k_seq_dim)
                    ], dim=self.k_seq_dim)
                    beacon_value = cat_tensor([
                        previous_beacon_value,
                        slice_tensor(concat_value, end=-raw_size_to_cache, dim=self.v_seq_dim)
                    ], dim=self.v_seq_dim)
                    raw_key = slice_tensor(concat_key, start=-raw_size_to_cache, dim=self.k_seq_dim)
                    raw_value = slice_tensor(concat_value, start=-raw_size_to_cache, dim=self.v_seq_dim)

                else:
                    # if we donot have raw activations, this means stride==window. we put all into beacon memory
                    beacon_key = cat_tensor([
                        previous_beacon_key,
                        key,
                    ], dim=self.k_seq_dim)
                    beacon_value = cat_tensor([
                        previous_beacon_value,
                        value,
                    ], dim=self.v_seq_dim)
                    raw_key = None
                    raw_value = None

            else:
                # [-beacon_size:] activations are from beacons, need to be accumulated
                # [-raw_cache_size-beacon_size:-beacon_size] raw activations will be cached; if they are shorter than raw_cache_size, part of the previous raw activations will also be kept

                beacon_key = cat_tensor([
                    previous_beacon_key,
                    slice_tensor(key, start=-beacon_size, dim=self.k_seq_dim)
                ], dim=self.k_seq_dim)
                beacon_value = cat_tensor([
                    previous_beacon_value,
                    slice_tensor(value, start=-beacon_size, dim=self.v_seq_dim)
                ], dim=self.v_seq_dim)

                if key.shape[self.k_seq_dim] < raw_size_to_cache + beacon_size:
                    concat_raw_key = cat_tensor([
                        previous_raw_key, 
                        slice_tensor(key, end=-beacon_size, dim=self.k_seq_dim)
                    ], dim=self.k_seq_dim)
                    concat_raw_value = cat_tensor([
                        previous_raw_value, 
                        slice_tensor(value, end=-beacon_size, dim=self.v_seq_dim)
                    ], dim=self.v_seq_dim)
                    raw_key = slice_tensor(concat_raw_key, start=-raw_size_to_cache, dim=self.k_seq_dim)
                    raw_value = slice_tensor(concat_raw_value, start=-raw_size_to_cache, dim=self.v_seq_dim)
                else:
                    # becomes None when raw_size_to_cache = 0
                    raw_key = slice_tensor(key, start=-raw_size_to_cache - beacon_size, end=-beacon_size, dim=self.k_seq_dim)
                    raw_value = slice_tensor(value, start=-raw_size_to_cache - beacon_size, end=-beacon_size, dim=self.v_seq_dim)

            self.beacon_activations[layer_idx] = (beacon_key, beacon_value)
            self.raw_activations[layer_idx] = (raw_key, raw_value)

        # NOTE: this is for debugging the resilience to different beacons
        # if self.step_idx == 2:
        #     print(self.get_memory_size())
        #     torch.save(self.beacon_activations, "beacon_activations")

    def update_loss(self, batch_loss, valid_token_num):
        """
        Accumulate loss for later perplexity computation and backward pass; past_key_values according to cache_method.
        """
        if self.batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self.batch_loss = batch_loss * valid_token_num
            self.valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self.batch_loss = self.batch_loss + batch_loss * valid_token_num
            self.valid_token_num = self.valid_token_num + valid_token_num

    def output(self, model_outputs):
        """
        Override loss with accumulated loss.
        """
        # override loss
        if self.batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self.batch_loss.sum() / self.valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self.batch_loss / self.valid_token_num
            if (self.valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self.valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss
            model_outputs["valid_token_num"] = self.valid_token_num

        # override last_hidden_states (used in generation)
        beacon_size = self._beacon_sizes[-1]
        # remove logits corresponding to beacon tokens
        if beacon_size > 0:
            model_outputs["logits"] = model_outputs["logits"][:, :-beacon_size]

        # print(f"process {dist.get_rank()}: loss {loss}")
        # print(f"process {dist.get_rank()}: beacon_sizes {self._beacon_sizes}")
        return model_outputs


def slice_tensor(x, start=None, end=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start == end:
        return None
    if dim == 2:
        if start is None and end is not None:
            return x[:, :, :end, ...]
        elif start is not None and end is None:
            return x[:, :, start:, ...]
        elif start is not None and end is not None:
            return x[:, :, start:end, ...]
    elif dim == 1:
        if start is None and end is not None:
            return x[:, :end, ...]
        elif start is not None and end is None:
            return x[:, start:, ...]
        elif start is not None and end is not None:
            return x[:, start:end, ...]
    else:
        raise NotImplementedError

def cat_tensor(list_of_tensors, dim=-1):
    list_of_tensors = [t for t in list_of_tensors if t is not None]
    if len(list_of_tensors) > 1:
        result = torch.cat(list_of_tensors, dim=dim)
    elif len(list_of_tensors) == 1:
        result = list_of_tensors[0]
    else:
        result = None
    return result

def softmax(x:np.ndarray, axis=-1, temperature=1):
    if isinstance(x, list):
        x = np.array(x)
    x = x / temperature
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def l1_norm(x):
    sum_x = sum(x)
    x = [y/sum_x for y in x]
    return x
