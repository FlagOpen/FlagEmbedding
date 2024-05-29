import os
import torch
import numpy as np
import torch.distributed as dist
from transformers.utils import logging
from transformers import AutoTokenizer
from itertools import cycle
from typing import List

logger = logging.get_logger(__name__)


class Memory(torch.nn.Module):
    def __init__(
        self, 
        model_config, 
        k_seq_dim:int=2, 
        v_seq_dim:int=2, 
    ):
        """Setup necessary attributes."""
        super().__init__()

        self.model_config = model_config

        # initialize necessary parameters
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_layers = model_config.num_hidden_layers
        self.max_position_embeddings = model_config.max_position_embeddings
        self.rng = np.random.default_rng(42)

        self.beacon_window = model_config.beacon_window
        self.beacon_stride = model_config.beacon_stride
        self.beacon_attn = model_config.beacon_attn
        self.beacon_ratio = model_config.beacon_ratio
        self.beacon_ratio_mix = model_config.beacon_ratio_mix
        self.beacon_param = model_config.beacon_param
        self.beacon_sink_size = model_config.beacon_sink_size
        self.beacon_attend_prev = model_config.beacon_attend_prev

        self.beacon_tokens = torch.zeros(1, dtype=torch.long) + model_config.vocab_size

        self.retrieval_method = model_config.retrieval_method
        self.retrieval_topk = model_config.retrieval_topk
        self.retrieval_key_length = model_config.retrieval_key_length
        self.retriever = None

        self._post_validation()
        self.reset()

    def _post_validation(self, verbose=True):
        assert self.beacon_window >= self.beacon_stride, f"Make sure the beacon_window {self.beacon_window} >= beacon_stride {self.beacon_stride}!"
        for ratio in self.beacon_ratio:
            assert ratio >= 0, f"Make sure all beacon ratios are greater than or equal to 0, found {self.beacon_ratio}!"
        assert self.beacon_attn in ["segmentation", "step-expansion", "full-coverage"], f"beacon_attn {self.beacon_attn} not implemented!"
        assert self.beacon_ratio_mix in ["instance-random", "step-random", "sequence", "join", "retrieval-tune"] or "adapt-" in self.beacon_ratio_mix, f"beacon_ratio_mix {self.beacon_ratio_mix} not implemented!"
        if self.retrieval_method is not None:
            assert self.beacon_ratio_mix == "join", f"Make sure the beacon_ratio_mix is join! Found {self.beacon_ratio_mix}."
        if self.beacon_ratio_mix == "join":
            # create another stream for moving gpu tensor to cpu
            # self.stream = torch.cuda.Stream()
            pass

        # set tokenizer and retriever here because this function will be called in `set` method
        if self.retrieval_method is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config._name_or_path, trust_remote_code=True)
            if self.retrieval_method == "bm25":
                from .modeling_retrieval import BM25Retriever
                if self.retriever is None or self.retriever.name != "bm25":
                    self.retriever = BM25Retriever()
            elif self.retrieval_method is not None:
                from .modeling_retrieval import DenseRetriever
                if self.retriever is None or self.retriever.name != self.retrieval_method:
                    self.retriever = DenseRetriever(encoder=self.retrieval_method)
        # elif self.retrieval_method == "m3":
        #     self.retriever = M3Retriever()
        self._cpu = torch.device("cpu")

        if verbose:
            info = f"applying activation beacon on {self.beacon_param} (the beacon embedding is initialized from {'bos' if self.model_config.beacon_embed_init == 'bos' else 'eos'} embedding), with window size {self.beacon_window}, stride {self.beacon_stride}, {self.beacon_attn} attention{' (attending to previous beacons)' if self.beacon_attend_prev else ' (no attending to previous beacons)'}, sink size {self.beacon_sink_size}, condensing ratio {self.beacon_ratio} (mixed by {self.beacon_ratio_mix}), {self.retrieval_method+' retrieval'+' top-'+str(self.retrieval_topk) + f' from {self.retrieval_key_length} key length corpus' if self.retrieval_method is not None else 'no retrieval'}..."
            logger.info(info)

    def set(self, verbose=True, **kwargs):
        if "beacon_ratio_mix" in kwargs and kwargs["beacon_ratio_mix"] == "join" and self.beacon_ratio_mix != "join":
            raise ValueError(f"You cannot switch beacon_ratio_mix from non-join strategy to join!")
        if self.beacon_ratio_mix == "join" and "beacon_ratio" in kwargs and sorted(kwargs["beacon_ratio"]) != sorted(self.beacon_ratio):
            raise ValueError(f"You cannot change beacon_ratio given beacon_ratio_mix=join!")
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._post_validation(verbose=verbose)

    def reset(self):
        """Initialize attributes for a new sequence."""
        # the cursor pointing to the start of the current window
        self._start_idx = 0
        # the cursor pointing to the end of the current window
        self._end_idx = 0
        # the beacon sizes of all strides
        self._total_beacon_sizes = []
        # the beacon ratios of all strides
        self._main_beacon_sizes = []
        # the loss per batch
        self._batch_loss = None
        # the valid token number per batch
        self._valid_token_num = None
        # the step index for processing the input_ids
        self._step_idx = 0

        # used in set_condensing_ratio
        self._ratio = None
        self._beacon_ratio_iter = None

        self.all_input_ids = torch.tensor([], dtype=torch.long)
        self.all_attention_mask = torch.tensor([], dtype=torch.long)
        if hasattr(self, "all_labels"):
            del self.all_labels

        # the raw activations of recent tokens
        self.raw_activations = [(None, None) for _ in range(self.num_layers)]
        # the attention sink activations
        self.sink_activations = [(None, None) for _ in range(self.num_layers)]

        # the beacon activations
        if self.beacon_ratio_mix == "join":
            self.l1_to_ln_beacon_activations = [
                [(None, None) for _ in range(self.num_layers)]
                for _ in self.beacon_ratio
            ]
        else:
            self.l1_to_ln_beacon_activations = [
                [(None, None) for _ in range(self.num_layers)]
            ]

        # used to control retrieval behavior
        self._do_retrieval = None
        self._retrieval_query = None
        self._retrieval_state = self.get_retrieval_state()

    def rewind(self, size=None, trim=False):
        """
        Rewind raw activations that have not been condensed yet.

        Args:
            trim: if true, the input_ids corresponding to the raw activations are trimmed.
        """
        raw_memory_size = self.get_memory_size()[1]
        if size is None:
            size = raw_memory_size
        assert size <= raw_memory_size, f"Make sure the rewind size ({size}) is smaller or equal to the raw memory size ({raw_memory_size})!"

        if size > 0:
            self._end_idx -= size
            for layer_idx, (key, value) in enumerate(self.raw_activations):
                key = slice_tensor(key, end=-size, dim=self.k_seq_dim)
                value = slice_tensor(value, end=-size, dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (key, value)

            if trim:
                self.all_input_ids = self.all_input_ids[:, :-size]
                self.all_attention_mask = self.all_attention_mask[:, :-size]
                if hasattr(self, "all_labels"):
                    self.all_labels = self.all_labels[:, :-size]

    @property
    def finish(self):
        is_finish = self._end_idx == self.all_sequence_length

        # print(f"{dist.get_rank()} Finish: {self._end_idx}, {self.all_sequence_length}")
        # if is_finish and hasattr(self, "stream"):
        #     self.stream.synchronize()
        return is_finish
    
    def get_memory_size(self):
        beacon_memory_size = 0
        raw_memory_size = 0
        if self.l1_to_ln_beacon_activations[0][0][0] is not None:
            beacon_memory_size += self.l1_to_ln_beacon_activations[0][0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        memory_size = beacon_memory_size + raw_memory_size
        return beacon_memory_size, raw_memory_size, memory_size

    def get_retrieval_state(self):
        retrieval_corpus_shape = tuple(self.all_input_ids[0, :self._start_idx].view(-1, self.retrieval_key_length)) if self.retrieval_key_length is not None else (0, 0)
        return (self.retrieval_method, self.retrieval_key_length, self.retrieval_topk, retrieval_corpus_shape, self._retrieval_query)

    def prepare(self, input_ids, attention_mask, labels):
        """
        Prepare inputs for the model. These inputs belong to the same sequence.
        """
        assert input_ids.shape[0] == 1, "Make sure the batch size is 1!"
        assert attention_mask is None or (attention_mask == 1).all(), "Make sure there is no padding!"

        if not hasattr(self, "_device"):
            self._device = input_ids.device

        # accumulate input_ids and attention_mask
        self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)
        self.all_sequence_length = self.all_input_ids.shape[1]

        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:].cpu(), torch.tensor([-100]).expand(labels.shape[0], 1)], dim=1)
            if not hasattr(self, "all_labels"):
                self.all_labels = labels
            else:
                self.all_labels = torch.cat([self.all_labels, labels], dim=1)
            assert self.all_input_ids.shape[1] == self.all_labels.shape[1], f"Found inconsistent all_input_ids {self.all_input_ids.shape} and all_labels {self.all_labels.shape}!"

        if self._do_retrieval is None:
            pass
        elif self._do_retrieval == "input":
            self._retrieval_query = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # NOTE: reset _do_retrieval because we only want to retrieval in the first generation step
            self._do_retrieval = "stale"
        elif self._do_retrieval == "stale":
            pass
        else:
            raise NotImplementedError(f"Retrieval with {self._do_retrieval} not implemented!")

    def set_condensing_ratio(self, start_idx, end_idx):
        """Choose a condensing ratio from self.beacon_ratio"""
        def filter_ratio(ratios, stride):
            valid_ratios = []
            for ratio in ratios:
                # stride must be bigger than condensing ratio because we there must be at least one beacon
                if stride < ratio:
                    continue
                # the stride must be evenly divisible by condensing ratio
                if ratio > 0 and (stride % ratio) != 0:
                    continue
                # when training, ratio=0 is valid if previous windows contain beacon or later windows contain beacon
                if ratio == 0 and self.training:
                    previous_has_zero = -1 in self._main_beacon_sizes
                    following_has_nonzero = (start_idx + stride + self.beacon_window) <= self.all_sequence_length
                    if previous_has_zero or (not following_has_nonzero):
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
            return [self.beacon_ratio[0]]

        ratio_mix = self.beacon_ratio_mix

        beacon_ratio = filter_ratio(self.beacon_ratio, self.beacon_stride)

        if ratio_mix == "instance-random":
            if self._ratio is None:
                beacon_ratio = self.rng.choice(beacon_ratio, size=1).tolist()
                self._ratio = beacon_ratio
            else:
                beacon_ratio = self._ratio

        elif ratio_mix == "step-random":
            beacon_ratio = self.rng.choice(beacon_ratio, size=1).tolist()
        
        elif ratio_mix == "sequence":
            if self._beacon_ratio_iter is None:
                self._beacon_ratio_iter = cycle(beacon_ratio)
            beacon_ratio = [next(self._beacon_ratio_iter)]
        
        elif ratio_mix == "join":
            beacon_ratio = beacon_ratio
        
        elif "adapt" in ratio_mix:
            if self._ratio is None:
                future_length = int(ratio_mix.split("-")[1])
                sequence_length = self.all_input_ids.shape[1] + future_length
                max_lengths = get_max_length(beacon_ratio)
                # ascendingly sort the max lengths
                valid_max_lengths_and_indices = [x for x in enumerate(max_lengths) if x[1] >= sequence_length]
                if len(valid_max_lengths_and_indices):
                    minimum_length_index = min(valid_max_lengths_and_indices, key=lambda x: x[1])[0]
                    # use the minimal possible length for this sequence (the smallest fold ratio)
                    beacon_ratio = [beacon_ratio[minimum_length_index]]
                else:
                    beacon_ratio = [max(beacon_ratio)]
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
        start_idx = self._start_idx
        # the end position of the current window w.r.t. the start of the current input sequence
        end_idx = start_idx + self.beacon_window

        # indicates if the current window is completely filled by raw activations and new tokens
        # we only append beacon tokens for full windows
        if end_idx > self.all_sequence_length:
            # the input is shorter than the initial window size
            end_idx = self.all_sequence_length
            is_full_window = False
        else:
            is_full_window = True

        # NOTE: in training, the entire sequence is input to the model at once
        # In the last window, we do not need to append beacons because they will not be used at all
        if self.training and end_idx == self.all_sequence_length:
            is_full_window = False

        # the real window size (remaining_size + new_token_size)
        window_size = end_idx - start_idx

        if is_full_window:
            beacon_stride = self.beacon_stride
            # a list of condensing ratios
            condensing_ratios = self.set_condensing_ratio(start_idx=start_idx, end_idx=end_idx)

            beacon_sizes = []
            for condensing_ratio in condensing_ratios:
                if condensing_ratio > 0:
                    # the stride must be evenly divisible by condensing_ratio
                    beacon_sizes.append(beacon_stride // condensing_ratio)
                else:
                    # the raw activations are used as beacon activations
                    beacon_sizes.append(-1)
            # forward start_idx and end_idx
            next_start_idx = start_idx + beacon_stride
            # how many raw activations to save
            raw_size_to_cache = end_idx - next_start_idx

            if hasattr(self, "_retrieval_span") and self.training:
                assert len(beacon_sizes) == 1
                # check if the current stride needs to store raw activations
                for s, e in self._retrieval_span:
                    if end_idx > s and start_idx < e and len(beacon_sizes) == 1:
                        beacon_sizes.append(-2)

                if len(self.l1_to_ln_beacon_activations) == 1:
                    assert self.get_memory_size()[0] == 0, f"Make sure there is no memory before enabling retrieval tuning!"
                    self.l1_to_ln_beacon_activations.append([(None, None) for _ in range(self.num_layers)])

        else:
            # no stride because the sequence has finished
            next_start_idx = start_idx
            # cache all recent raw activations to be used in the next window
            raw_size_to_cache = window_size
            beacon_sizes = [0]
            condensing_ratios = [0]

        total_beacon_size = sum(s for s in beacon_sizes if s >= 0)

        # generate memory (memory_length = old_beacon_size + beacon_size * condensing_ratio + raw_cache_size)
        # TODO: add retrieval here
        past_key_values = []
        default_compose_memory = True

        # TODO: add batch size
        if self._do_retrieval is not None and self.retrieval_method is not None:
            assert self.all_input_ids.shape[0] == 1, f"Make sure batch_size is 1 when enabling retrieval!"

            # perform retrieval
            default_compose_memory = False

            retrieval_state = self.get_retrieval_state()
            # NOTE: do retrieval when the state changes
            if retrieval_state != self._retrieval_state:
                corpus = self.all_input_ids[0, :start_idx].view(-1, self.retrieval_key_length)
                corpus = self.tokenizer.batch_decode(corpus, skip_special_tokens=True)

                self.retriever.remove_all()
                self.retriever.add(corpus)

                retrieval_scores, retrieval_indices = self.retriever.search(self._retrieval_query, hits=self.retrieval_topk)
                # print(retrieval_scores, retrieval_indices)
                # NOTE: important to sort the indices so that adjacent indices can be merged
                score_indice_pairs = sorted(zip(retrieval_scores[0], retrieval_indices[0]), key=lambda x: x[1])

                self._retrieval_corpus = corpus
                self._retrieval_scores = [p[0] for p in score_indice_pairs]
                self._retrieval_indices = [p[1] for p in score_indice_pairs]
                self._retrieval_state = retrieval_state

                max_token_idx = self.retriever.num_keys * self.retrieval_key_length

                main_activations = self.l1_to_ln_beacon_activations[0]
                augment_activations = self.l1_to_ln_beacon_activations[-1]
                main_condensing_ratio = self.beacon_ratio[0]
                augment_condensing_ratio = self.beacon_ratio[1]
                # 0 means no condensing, however it triggers zero division error
                # use 1 instead
                if augment_condensing_ratio == 0:
                    augment_condensing_ratio = 1

                retrieval_token_span = []
                prev_token_end = None
                for idx in self._retrieval_indices:
                    # NOTE: concat the previous and the next interval
                    token_start = idx * self.retrieval_key_length
                    token_end = min((idx + 1) * self.retrieval_key_length, self.retriever.num_keys * self.retrieval_key_length)
                    if token_start == prev_token_end:
                        retrieval_token_span[-1][1] = token_end
                    else:
                        retrieval_token_span.append([token_start, token_end])
                    prev_token_end = token_end

                # get the non-retrieved token span (their activations will be preserved)
                non_retrieval_token_span = []
                if retrieval_token_span[0][0] > 0:
                    non_retrieval_token_span.append([0, retrieval_token_span[0][0]])
                for j in range(len(retrieval_token_span)):
                    non_retrieval_token_span.append([None, None])
                    if j == len(retrieval_token_span) - 1:
                        if retrieval_token_span[j][1] < max_token_idx:
                            # NOTE: replace the non-retrieved activations with retrieved ones
                            non_retrieval_token_span.append([retrieval_token_span[j][1], max_token_idx])

                            # NOTE: append the retrieved activations in front of the non-retrieved ones
                            # non_retrieval_token_span.append([retrieval_token_span[j][0], max_token_idx])
                    else:
                        # NOTE: replace the non-retrieved activations with retrieved ones
                        non_retrieval_token_span.append([retrieval_token_span[j][1], retrieval_token_span[j + 1][0]])

                        # NOTE: append the retrieved activations in front of the non-retrieved ones
                        # non_retrieval_token_span.append([retrieval_token_span[j][0], retrieval_token_span[j + 1][0]])

                # non_retrieved_token_span = [[None, None] for _ in retrieved_token_span]

                retrieval_beacon_span = [(s // augment_condensing_ratio, e // augment_condensing_ratio) for s, e in retrieval_token_span]
                non_retrieval_beacon_span = [(s // main_condensing_ratio if s is not None else None, e // main_condensing_ratio if e is not None else None) for s, e in non_retrieval_token_span]

                # compose beacon activations based on retrieved and non-retrieved activations
                self._beacon_activations = interleave_activations(
                    main_activations=main_activations,
                    augment_activations=augment_activations,
                    main_spans=non_retrieval_beacon_span,
                    augment_spans=retrieval_beacon_span,
                    k_seq_dim=self.k_seq_dim,
                    v_seq_dim=self.v_seq_dim,
                    device=self._device
                )

                # print(f"Memory Length: {unified_key.shape[self.k_seq_dim]}")

        if hasattr(self, "_retrieval_span") and self.training and end_idx == self.all_sequence_length:
            # switch activations at the last stride
            default_compose_memory = False

            main_activations = self.l1_to_ln_beacon_activations[0]
            augment_activations = self.l1_to_ln_beacon_activations[1]

            retrieval_token_span = self._retrieval_span
            # get the non-retrieved token span (their activations will be preserved)
            non_retrieval_token_span = []
            if retrieval_token_span[0][0] > 0:
                non_retrieval_token_span.append([0, retrieval_token_span[0][0]])
            for j in range(len(retrieval_token_span)):
                non_retrieval_token_span.append([None, None])
                if j == len(retrieval_token_span) - 1:
                    if retrieval_token_span[j][1] < start_idx:
                        non_retrieval_token_span.append([retrieval_token_span[j][1], start_idx])
                else:
                    non_retrieval_token_span.append([retrieval_token_span[j][1], retrieval_token_span[j + 1][0]])

            retrieval_beacon_span = [(s, e) for s, e in retrieval_token_span]
            non_retrieval_beacon_span = []
            for s, e in non_retrieval_token_span:
                if s is not None:
                    window_idx = s // self.beacon_window
                    # step_idx equals window_idx because when training we input the entire sequence at once
                    main_condensing_ratio = self.beacon_window // self._main_beacon_sizes[window_idx]
                    non_retrieval_beacon_span.append((s // main_condensing_ratio, e // main_condensing_ratio))
                else:
                    non_retrieval_beacon_span.append((None, None))

            # if dist.get_rank() == 0:
            #     print(retrieval_token_span)
            #     print(retrieval_beacon_span)
            #     print(non_retrieval_token_span)
            #     print(non_retrieval_beacon_span)

            self._beacon_activations = interleave_activations(
                main_activations=main_activations,
                augment_activations=augment_activations,
                main_spans=non_retrieval_beacon_span,
                augment_spans=retrieval_beacon_span,
                k_seq_dim=self.k_seq_dim,
                v_seq_dim=self.v_seq_dim,
                device=self._device
            )

        if default_compose_memory:
            self._beacon_activations = self.l1_to_ln_beacon_activations[0]

        for layer_idx in range(self.num_layers):
            sink_key, sink_value = self.sink_activations[layer_idx]
            beacon_key, beacon_value = self._beacon_activations[layer_idx]
            raw_key, raw_value = self.raw_activations[layer_idx]

            key = cat_tensor([
                sink_key, beacon_key, raw_key,
            ], dim=self.k_seq_dim)
            value = cat_tensor([
                sink_value, beacon_value, raw_value,
            ], dim=self.v_seq_dim)

            layer_past_key_values = (key, value, beacon_sizes, total_beacon_size, raw_size_to_cache, window_size)
            past_key_values.append(layer_past_key_values)

            # if key is not None:
            #     print(f"Memory Length: {key.shape[self.k_seq_dim]}")

        # streamingly add new input_ids
        input_ids = self.all_input_ids[:, self._end_idx: end_idx].to(self._device)
        attention_mask = self.all_attention_mask[:, self._end_idx: end_idx].to(self._device)
        if hasattr(self, "all_labels"):
            labels = self.all_labels[:, self._end_idx: end_idx].to(self._device)
        else:
            labels = None
        batch_size = input_ids.shape[0]

        # append beacons if necessary
        if is_full_window:
            if total_beacon_size > 0:
                input_ids = torch.cat([input_ids, self.beacon_tokens.expand(batch_size, total_beacon_size).to(input_ids.device, dtype=input_ids.dtype)], dim=1)
                # NOTE: prepend beacon_memory_size 1 to attention_mask because we have past_key_values
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, total_beacon_size)], dim=1)
                if labels is not None:
                    labels = torch.cat([labels, labels.new_zeros(batch_size, total_beacon_size) - 100], dim=1)

        # prepend 1 to attention mask for previous memory
        first_key = past_key_values[0][0]
        memory_size = first_key.shape[self.k_seq_dim] if first_key is not None else 0
        if memory_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, memory_size), attention_mask], dim=1)

        # involked in self.output()
        self._total_beacon_sizes.append(total_beacon_size)
        # involked in self.set_condensing_ratio
        self._main_beacon_sizes.append(beacon_sizes[0])

        # update end_idx
        self._start_idx = next_start_idx
        self._end_idx = end_idx
        self._step_idx += 1

        # print("****************************************")
        # if is_full_window:
        #     print(f"stride:             {beacon_stride}")
        #     print(f"condensing ratios:  {condensing_ratios}")
        #     print(f"beacon_sizes:       {beacon_sizes}")
        # print(f"input_ids:          {input_ids.shape}")
        # print(f"start_idx:          {start_idx}")
        # print(f"next_start_idx:     {next_start_idx}")
        # print(f"end_idx:            {end_idx}")
        # x = input()
        # if x == "s":
        #     return
        
        return input_ids, attention_mask, past_key_values, labels

    def update_memory(self, past_key_values):
        """
        Accumulate beacon activations and raw activations.
        """
        for layer_idx, (key, value, beacon_sizes, total_beacon_size, raw_size_to_cache, window_size) in enumerate(past_key_values):
            # NOTE: the past_key_values are incrementally returned (only the new keys and values are returned)

            # key/value: (num_layer, 2, batch_size, num_head, new_seq_len, head_dim)
            # beacon_size: how many beacon activations are in key and value
            # raw_size_to_cache: how many raw activations should be kept

            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]

            if self._step_idx == 1:
                # save the sink activations
                # NOTE: we do not slice the key/value activations, which may cause duplication when beacon_ratio=-1 for the first window, but it's okay
                self.sink_activations[layer_idx] = [
                    slice_tensor(key, end=self.beacon_sink_size, dim=self.k_seq_dim),
                    slice_tensor(value, end=self.beacon_sink_size, dim=self.v_seq_dim),
                ]

            if beacon_sizes == [0]:
                # this means the current input does not fulfill a window
                # thus, the key and value are all raw activations, and we accumulate them until the window is fulfilled
                assert raw_size_to_cache == window_size
                raw_key = cat_tensor([
                    previous_raw_key,
                    key
                ], dim=self.k_seq_dim)
                raw_value = cat_tensor([
                    previous_raw_value, 
                    value
                ], dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (raw_key, raw_value)

            else:
                for beacon_size_idx, beacon_size in enumerate(beacon_sizes):
                    # NOTE: use the correct previous_beacon_key and value!
                    previous_beacon_key, previous_beacon_value = self.l1_to_ln_beacon_activations[beacon_size_idx][layer_idx]
                    
                    # if beacon_size_idx == 0:
                    #     ctx_manager = nullcontext()
                    # else:
                    #     ctx_manager = torch.cuda.stream(self.stream)                    
                    # FIXME: only the first iteration works...
                    # with ctx_manager:

                    beacon_key, beacon_value, raw_key, raw_value = self._extract_beacon_and_raw_memory(key, value, previous_beacon_key, previous_beacon_value, previous_raw_key, previous_raw_value, raw_size_to_cache, total_beacon_size, beacon_sizes, beacon_size_idx)

                    self.l1_to_ln_beacon_activations[beacon_size_idx][layer_idx] = (beacon_key, beacon_value)
                    if beacon_size_idx == 0:
                        self.raw_activations[layer_idx] = (raw_key, raw_value)
                        
                    # if beacon_size_idx != 0:
                    #     print(self.stream.query())

    def update_loss(self, batch_loss, valid_token_num):
        """
        Accumulate loss for later perplexity computation and backward pass; past_key_values according to cache_method.
        """
        # print(f"process {dist.get_rank()}: valid_token_num: {valid_token_num}; loss {batch_loss}")
        if self._batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self._batch_loss = batch_loss * valid_token_num
            self._valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self._batch_loss = self._batch_loss + batch_loss * valid_token_num
            self._valid_token_num = self._valid_token_num + valid_token_num

    def output(self, model_outputs):
        """
        Override loss with accumulated loss.
        """
        # override loss
        if self._batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self._batch_loss.sum() / self._valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self._batch_loss / self._valid_token_num
            if (self._valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self._valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss
            model_outputs["valid_token_num"] = self._valid_token_num

        # override last_hidden_states (used in generation)
        beacon_size = self._total_beacon_sizes[-1]
        # remove logits corresponding to beacon tokens
        if beacon_size > 0:
            model_outputs["logits"] = model_outputs["logits"][:, :-beacon_size]

        return model_outputs

    def _extract_beacon_and_raw_memory(self, key, value, previous_beacon_key, previous_beacon_value, previous_raw_key, previous_raw_value, raw_size_to_cache, total_beacon_size, beacon_sizes, beacon_size_idx):
        """Extract beacon and raw memory from the returned key and value. The raw memory is computed only if the beacon_size_idx == 0."""
        beacon_size = beacon_sizes[beacon_size_idx]
        # NOTE: ignore -1
        previous_beacon_size = sum(x for x in beacon_sizes[:beacon_size_idx] if x > 0)

        if previous_beacon_key is not None:
            target_device = previous_beacon_key.device
        else:
            if beacon_size_idx == 0:
                target_device = self._device
            else:
                target_device = self._cpu

        if beacon_size == -1:
            actual_beacon_size = self.beacon_window - raw_size_to_cache

            # the raw activations are used as beacon activations
            concat_raw_key = cat_tensor([
                previous_raw_key, 
                key
            ], dim=self.k_seq_dim)
            concat_raw_value = cat_tensor([
                previous_raw_value, 
                value
            ], dim=self.v_seq_dim)

            beacon_key = cat_tensor([
                previous_beacon_key,
                slice_tensor(concat_raw_key, end=actual_beacon_size, dim=self.k_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.k_seq_dim)
            beacon_value = cat_tensor([
                previous_beacon_value,
                slice_tensor(concat_raw_value, end=actual_beacon_size, dim=self.v_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.v_seq_dim)

            if beacon_size_idx == 0:
                raw_key = slice_tensor(concat_raw_key, start=actual_beacon_size, end=self.beacon_window, dim=self.k_seq_dim)
                raw_value = slice_tensor(concat_raw_value, start=actual_beacon_size, end=self.beacon_window, dim=self.v_seq_dim)
        
        elif beacon_size == -2:
            assert beacon_size_idx == 1, f"beacon_size == -2 is only applicable with retrieval-oriented tuning!"
            # the beacon_key and value will be filled with 0
            actual_beacon_size = self.beacon_window - raw_size_to_cache

            # the raw activations are used as beacon activations
            concat_raw_key = cat_tensor([
                previous_raw_key, 
                key
            ], dim=self.k_seq_dim)
            concat_raw_value = cat_tensor([
                previous_raw_value, 
                value
            ], dim=self.v_seq_dim)

            key_shape = list(key.shape)
            value_shape = list(value.shape)
            key_shape[self.k_seq_dim] = actual_beacon_size
            value_shape[self.v_seq_dim] = actual_beacon_size

            beacon_key = cat_tensor([
                previous_beacon_key,
                key.new_zeros(key_shape, device=target_device),
            ], dim=self.k_seq_dim)
            beacon_value = cat_tensor([
                previous_beacon_value,
                value.new_zeros(value_shape, device=target_device),
            ], dim=self.v_seq_dim)

        else:
            # [-beacon_size:] activations are from beacons, need to be accumulated
            # [-raw_cache_size-beacon_size:-beacon_size] raw activations will be cached; if they are shorter than raw_cache_size, part of the previous raw activations will also be kept
            
            beacon_start_idx = - total_beacon_size + previous_beacon_size
            beacon_end_idx = beacon_start_idx + beacon_size

            # NOTE: avoid end=0 for slicing
            if beacon_end_idx == 0:
                beacon_end_idx = None
            
            beacon_key = cat_tensor([
                previous_beacon_key,
                slice_tensor(key, start=beacon_start_idx, end=beacon_end_idx, dim=self.k_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.k_seq_dim)
            beacon_value = cat_tensor([
                previous_beacon_value,
                slice_tensor(value, start=beacon_start_idx, end=beacon_end_idx, dim=self.v_seq_dim).to(target_device, non_blocking=True)
            ], dim=self.v_seq_dim)

            # the raw activations are only updated once
            if beacon_size_idx == 0:
                if key.shape[self.k_seq_dim] < raw_size_to_cache + beacon_size:
                    concat_raw_key = cat_tensor([
                        previous_raw_key, 
                        key
                    ], dim=self.k_seq_dim)
                    concat_raw_value = cat_tensor([
                        previous_raw_value, 
                        value
                    ], dim=self.v_seq_dim)
                    raw_key = slice_tensor(concat_raw_key, start=self.beacon_window - raw_size_to_cache, end=self.beacon_window, dim=self.k_seq_dim)
                    raw_value = slice_tensor(concat_raw_value, start=self.beacon_window - raw_size_to_cache, end=self.beacon_window, dim=self.v_seq_dim)
                else:
                    # becomes None when raw_size_to_cache = 0
                    raw_key = slice_tensor(key, start=beacon_start_idx - raw_size_to_cache, end=beacon_start_idx, dim=self.k_seq_dim)
                    raw_value = slice_tensor(value, start=beacon_start_idx - raw_size_to_cache, end=beacon_start_idx, dim=self.v_seq_dim)

        if beacon_size_idx == 0:
            return beacon_key, beacon_value, raw_key, raw_value
        else:
            # NOTE: only l1 beacon activations are kept on GPU
            return beacon_key.detach().to(target_device, non_blocking=True), beacon_value.detach().to(target_device, non_blocking=True), None, None
            # return beacon_key, beacon_value, None, None


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

def slice_activations(activations, start=None, end=None, k_seq_dim=2, v_seq_dim=2):
    new_activations = []
    for key, value in activations:
        new_key = slice_tensor(key, start=start, end=end, dim=k_seq_dim)
        new_value = slice_tensor(value, start=start, end=end, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def cat_activations(list_of_activations, k_seq_dim=2, v_seq_dim=2):
    assert all(len(x) == len(list_of_activations[0]) for x in list_of_activations), f"Make sure all activations have the same number of layers! Found {[len(x) for x in list_of_activations]}."

    new_activations = []
    for layer_idx in range(len(list_of_activations[0])):
        keys = [x[layer_idx][0] for x in list_of_activations]
        values = [x[layer_idx][1] for x in list_of_activations]

        new_key = cat_tensor(keys, dim=k_seq_dim)
        new_value = cat_tensor(values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def interleave_activations(main_activations, augment_activations, main_spans, augment_spans, k_seq_dim=2, v_seq_dim=2, device=torch.device("cuda")):
    """ Interleave main_activations and augment_activations according to main_span and augment_span.

    Args:
        main_span: a list of tuples (start_idx, end_idx). when start_idx and end_idx is None, the augment_activations will be plugged in.
        augment_span: a list of tuples (start_idx, end_idx)
    """
    assert len(main_activations) == len(augment_activations) , f"Make sure main and augment activations have the same number of layers! Found {len(main_activations)} and {len(augment_activations)}!"
    assert sum(x[0] is None and x[1] is None for x in main_spans) == len(augment_spans), f"Make sure the number of slots for augmentation (start_idx=None and end_idx=None in main_spans) matches the number of augmentations. Found {sum(x for x in main_spans if x[0] is None and x[1] is None)} slots but {len(augment_spans)} augmentations!"

    new_activations = []
    for layer_idx in range(len(main_activations)):
        main_key, main_value = main_activations[layer_idx]
        augment_key, augment_value = augment_activations[layer_idx]

        sliced_keys = []
        sliced_values = []

        augment_idx = 0
        for start, end in main_spans:
            if start is None and end is None:
                # this means the augment key/value should be plugged in
                augment_start, augment_end = augment_spans[augment_idx]
                sliced_key = slice_tensor(
                    augment_key, 
                    start=augment_start, 
                    end=augment_end,
                    dim=k_seq_dim
                ).to(device)
                sliced_value = slice_tensor(
                    augment_value, 
                    start=augment_start, 
                    end=augment_end,
                    dim=v_seq_dim
                ).to(device)

            else:
                sliced_key = slice_tensor(
                    main_key, 
                    start=start, 
                    end=end,
                    dim=k_seq_dim
                )
                sliced_value = slice_tensor(
                    main_value, 
                    start=start, 
                    end=end,
                    dim=v_seq_dim
                )

            sliced_keys.append(sliced_key)
            sliced_values.append(sliced_value)

        new_key = cat_tensor(sliced_keys, dim=k_seq_dim)
        new_value = cat_tensor(sliced_values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])

    return new_activations

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