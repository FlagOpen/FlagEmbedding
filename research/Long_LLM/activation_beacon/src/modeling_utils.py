import math
import torch
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Mapping, Optional, Tuple
from accelerate import Accelerator
from collections import defaultdict
from transformers.modeling_outputs import BaseModelOutputWithPast


def optional_grad_ctx(with_grad=False):
    if with_grad:
        return nullcontext()
    else:
        return torch.no_grad()

def move_to_device(data, device):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    else:
        return data

def get_shifted_labels(input_ids):
    if isinstance(input_ids, torch.Tensor):
        labels = input_ids.clone()
        labels = torch.cat([labels[:, 1:], labels.new_zeros((input_ids.shape[0], 1)) - 100], dim=-1)
    elif isinstance(input_ids, list) and isinstance(input_ids[0], int):
        labels = input_ids.copy()
        labels = labels[1:] + [-100]
    elif isinstance(input_ids, list) and isinstance(input_ids[0], list):
        labels = input_ids.copy()
        for i, label in enumerate(labels):
            labels[i] = labels[i][1:] + [-100]
    else:
        raise NotImplementedError
    return labels

def compute_loss(logits, labels, shift=False):
    """
    Returns:
        token_loss: batch_size, seq_length
    """
    if shift:
        labels = get_shifted_labels(labels)

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # NOTE: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len

    # print(token_loss)

    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, token_loss


@torch.no_grad()
def evaluate_perplexity(model, dataloader, accelerator:Optional[Accelerator]=None):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    all_loss = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # the seq id
        index = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        output = model(**x)

        valid_token_num = (x["labels"] != -100).sum(-1)

        # NOTE: we need the loss for each element in the batch for accurate computation, because the number of valid tokens may differ among elements
        if hasattr(output, "batch_loss"):
            # output from our model has batch_loss by default
            batch_loss = output.batch_loss
        else:
            # output from other models does not
            loss, batch_loss, token_loss = compute_loss(output.logits, x["labels"], shift=True)

        index = index.tolist()
        batch_loss = batch_loss.tolist()
        valid_token_num = valid_token_num.tolist()

        if accelerator is not None and accelerator.num_processes > 1:
            # num_device * batch_size
            index = accelerator.gather_for_metrics(index)
            batch_loss = accelerator.gather_for_metrics(batch_loss)
            valid_token_num = accelerator.gather_for_metrics(valid_token_num)

        for _id, _loss, _num in zip(index, batch_loss, valid_token_num):
            # loss times num is the total loss of all valid tokens
            all_loss[_id].append((_loss * _num, _num))

    all_loss = dict(all_loss)
    for _id, loss_and_num in all_loss.items():
        # sum up the loss for all valid tokens in the entire sequence, and divide the number of valid tokens
        all_loss[_id] = sum([x[0] for x in loss_and_num]) / sum(x[1] for x in loss_and_num)
    
    # average across then take exp
    perplexity = math.exp(sum(all_loss.values()) / len(all_loss))
    return perplexity


@torch.no_grad()
def evaluate_generation(model, dataloader, accelerator:Optional[Accelerator]=None, tokenizer=None, return_new_tokens_only=True, **generation_config):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    all_indices = []
    all_outputs = []

    index = 0
    
    for i, x in enumerate(tqdm(dataloader, desc="Computing Generation")):
        # if i > 3:
        #     break
        
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # length is used to group training data, no use here
        length = x.pop("length", None)

        # if indices are None, we use batch size
        indices = x.pop("index", None)
        if indices is None:
            indices = list(range(index, index + x['input_ids'].shape[0]))
            index += x['input_ids'].shape[0]
        else:
            indices = indices.tolist()

        outputs = model.generate(**x, **generation_config)
        if return_new_tokens_only:
            start_idx = x["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if accelerator is not None and accelerator.num_processes > 1:
            outputs = accelerator.gather_for_metrics(outputs)
            indices = accelerator.gather_for_metrics(indices)

        outputs = outputs
        indices = indices
        all_indices.extend(indices)
        all_outputs.extend(outputs)

    return all_indices, all_outputs


@torch.no_grad()
def evaluate_nll(model, dataloader, accelerator:Optional[Accelerator]=None):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    all_loss = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # the seq id
        index = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        output = model(**x)

        valid_token_num = (x["labels"] != -100).sum()

        # NOTE: we need the loss for each element in the batch for accurate computation, because the number of valid tokens may differ among elements
        if hasattr(output, "batch_loss"):
            # output from our model has batch_loss by default
            batch_loss = output.batch_loss
        else:
            # output from other models does not
            loss, batch_loss, token_loss = compute_loss(output.logits, x["labels"], shift=True)

        if accelerator is not None and accelerator.num_processes > 1:
            # num_device * batch_size
            index = accelerator.gather_for_metrics(index)
            batch_loss = accelerator.gather_for_metrics(batch_loss)
            valid_token_num = accelerator.gather_for_metrics(valid_token_num)

        for _id, _loss in zip(index.tolist(), batch_loss.tolist()):
            # loss times num is the total loss of all valid tokens
            all_loss[_id].append(_loss)

    return all_loss


@dataclass
class ModelOutput(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    batch_loss: Optional[torch.FloatTensor] = None
    token_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



########## Various RoPE Scaling Methods Below (wrap the encoding process within the module for convenience) ##########

def get_rope(head_dim, base, max_position_embeddings, rope_scaling=None):
    """
    Get rope module. {native, linear scaling, dynamic ntk scaling, yarn scaling, llama3 scaling}
    """
    if rope_scaling is None:
        rope = RotaryEmbedding(
            dim=head_dim,
            base=base,
            max_position_embeddings=max_position_embeddings,
        )
    else:
        scaling_type = rope_scaling["type"]
        scaling_factor = rope_scaling["factor"]
        if scaling_type == "linear":
            rope = LinearScalingRotaryEmbedding(
                dim=head_dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "dynamic":
            rope = DynamicNTKScalingRotaryEmbedding(
                dim=head_dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "yarn":
            rope = YarnRotaryEmbedding(
                dim=head_dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "yarn-t":
            rope = YarnDynamicTemperatureRotaryEmbedding(
                dim=head_dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "yarn-t-logn":
            rope = YarnDynamicTemperatureLogNRotaryEmbedding(
                dim=head_dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "llama3":
            rope = Llama3RotaryEmbedding(
                dim=head_dim,
                base=base,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=scaling_factor,
                original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 8192),
                low_freq_factor=rope_scaling.get("low_freq_factor", 1),
                high_freq_factor=rope_scaling.get("high_freq_factor", 4),
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    
    return rope


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device, dtype=k.dtype)

        # batch_size, 1, key_len, head_dim
        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class YarnRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, beta_slow=2, beta_fast=128):
        super().__init__()

        self.base = base
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.max_position_embeddings = max_position_embeddings

        self._set_cos_sin_cache(
            seq_len=math.ceil(max_position_embeddings * scaling_factor), device=device, dtype=torch.get_default_dtype()
        )

    def _get_factor(self):
        # the dimension whose index is smaller than fast_dim rotates more than beta_fast
        fast_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_fast)) / math.log(self.base))
        fast_dim = max(math.floor(fast_dim), 0)
        # the dimension whose index is bigger than slow_dim rotates less than beta_slow
        slow_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_slow)) / math.log(self.base))
        slow_dim = min(math.ceil(slow_dim), self.dim - 1)

        if fast_dim == slow_dim:
            slow_dim += 0.001

        # NOTE: very important to use full precision here so that the factor is correct
        dim_arange = torch.arange(0, self.dim // 2, dtype=torch.float32)
        dim_factor = (dim_arange - fast_dim) / (slow_dim - fast_dim)
        dim_factor = torch.clamp(dim_factor, 0, 1)

        # align with the paper notation
        return (1 - dim_factor)

    def _get_temperature(self):
        if self.scaling_factor <= 1:
            return 1.0
        return 0.07 * math.log(self.scaling_factor) + 1.0
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        dim_arange = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        # dim / 2
        freq = self.base ** dim_arange
        theta = 1 / freq
        interleave_theta = theta / self.scaling_factor

        factor = self._get_factor().to(device)
        yarn_theta = factor * theta + (1 - factor) * interleave_theta
        self.register_buffer("inv_freq", yarn_theta, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # get attention temperature
        temperature = self._get_temperature()

        self.register_buffer("cos_cached", emb.cos() * temperature, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * temperature, persistent=False)
        self.max_seq_len_cached = seq_len
    
    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.scaling_factor = seq_len / self.max_position_embeddings
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device, dtype=k.dtype)

        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed


class YarnDynamicTemperatureRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, beta_slow=2, beta_fast=128):
        super().__init__()

        self.base = base
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.max_position_embeddings = max_position_embeddings

        self._set_cos_sin_cache(
            seq_len=math.ceil(max_position_embeddings * scaling_factor), device=device, dtype=torch.get_default_dtype()
        )

    def _get_factor(self):
        # the dimension whose index is smaller than fast_dim rotates more than beta_fast
        fast_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_fast)) / math.log(self.base))
        fast_dim = max(math.floor(fast_dim), 0)
        # the dimension whose index is bigger than slow_dim rotates less than beta_slow
        slow_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_slow)) / math.log(self.base))
        slow_dim = min(math.ceil(slow_dim), self.dim - 1)

        if fast_dim == slow_dim:
            slow_dim += 0.001

        # NOTE: very important to use full precision here so that the factor is correct
        dim_arange = torch.arange(0, self.dim // 2, dtype=torch.float32)
        dim_factor = (dim_arange - fast_dim) / (slow_dim - fast_dim)
        dim_factor = torch.clamp(dim_factor, 0, 1)

        # align with the paper notation
        return (1 - dim_factor)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        dim_arange = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        # dim / 2
        freq = self.base ** dim_arange
        theta = 1 / freq
        interleave_theta = theta / self.scaling_factor

        factor = self._get_factor().to(device)
        yarn_theta = factor * theta + (1 - factor) * interleave_theta
        self.register_buffer("inv_freq", yarn_theta, persistent=False)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # NOTE: get attention temperature that will be applied on the query vector
        # temperature = torch.log(positions + 1) / math.log(self.max_position_embeddings)
        temperature = (0.07 * torch.log((positions + 1) / self.max_position_embeddings) + 1) ** 2
        temperature[:self.max_position_embeddings] = 1
        self.register_buffer("temperature", temperature.unsqueeze(1), persistent=False)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len
    
    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.scaling_factor = seq_len / self.max_position_embeddings
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device, dtype=k.dtype)

        # batch_size, 1, key_len, head_dim
        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_position_ids = position_ids[:, -q.shape[2]:]
        temperature = self.temperature[q_position_ids].to(dtype=k.dtype).unsqueeze(1)
        q_cos = q_cos * temperature
        q_sin = q_sin * temperature

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed


class YarnDynamicTemperatureLogNRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, beta_slow=2, beta_fast=128):
        super().__init__()

        self.base = base
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.max_position_embeddings = max_position_embeddings

        self._set_cos_sin_cache(
            seq_len=math.ceil(max_position_embeddings * scaling_factor), device=device, dtype=torch.get_default_dtype()
        )

    def _get_factor(self):
        # the dimension whose index is smaller than fast_dim rotates more than beta_fast
        fast_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_fast)) / math.log(self.base))
        fast_dim = max(math.floor(fast_dim), 0)
        # the dimension whose index is bigger than slow_dim rotates less than beta_slow
        slow_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_slow)) / math.log(self.base))
        slow_dim = min(math.ceil(slow_dim), self.dim - 1)

        if fast_dim == slow_dim:
            slow_dim += 0.001

        # NOTE: very important to use full precision here so that the factor is correct
        dim_arange = torch.arange(0, self.dim // 2, dtype=torch.float32)
        dim_factor = (dim_arange - fast_dim) / (slow_dim - fast_dim)
        dim_factor = torch.clamp(dim_factor, 0, 1)

        # align with the paper notation
        return (1 - dim_factor)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        dim_arange = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        # dim / 2
        freq = self.base ** dim_arange
        theta = 1 / freq
        interleave_theta = theta / self.scaling_factor

        factor = self._get_factor().to(device)
        yarn_theta = factor * theta + (1 - factor) * interleave_theta
        self.register_buffer("inv_freq", yarn_theta, persistent=False)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # NOTE: get attention temperature that will be applied on the query vector
        temperature = torch.log(positions + 1) / math.log(self.max_position_embeddings)
        # temperature = (0.07 * torch.log((positions + 1) / self.max_position_embeddings) + 1) ** 2
        temperature[:self.max_position_embeddings] = 1
        self.register_buffer("temperature", temperature.unsqueeze(1), persistent=False)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len
    
    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.scaling_factor = seq_len / self.max_position_embeddings
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device, dtype=k.dtype)

        # batch_size, 1, key_len, head_dim
        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_position_ids = position_ids[:, -q.shape[2]:]
        temperature = self.temperature[q_position_ids].to(dtype=k.dtype).unsqueeze(1)
        q_cos = q_cos * temperature
        q_sin = q_sin * temperature

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed


class Llama3RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000, device=None, scaling_factor=1.0, original_max_position_embeddings=8192, low_freq_factor=1, high_freq_factor=4):
        super().__init__()

        self.base = base
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.max_position_embeddings = max(max_position_embeddings, int(original_max_position_embeddings * scaling_factor))
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim))
        low_freq_wavelen = self.original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = self.original_max_position_embeddings / high_freq_factor
        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scaling_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (self.original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scaling_factor + smooth * freq)
        inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device)

        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed
