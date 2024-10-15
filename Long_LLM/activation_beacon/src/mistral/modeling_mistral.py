# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.integrations import is_deepspeed_zero3_enabled
from .configuration_mistral import MistralConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

from ..modeling_beacon import Memory
from ..modeling_utils import optional_grad_ctx, compute_loss, get_rope, ModelOutput


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"

# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.MistralRMSNorm with Mistral->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Mistral
class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = get_rope(self.head_dim, config.rope_theta, config.max_position_embeddings, getattr(config, "rope_scaling", None))

        # NOTE: add extra parameters for beacon tokens
        # skip post initialization to speed up loading
        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.q_proj.bias is not None)
            # NOTE: initialize the beacon parameters as zero
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in config.beacon_param:
            self.beacon_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.k_proj.bias is not None)
            self.beacon_k_proj.weight.data.zero_()
            self.beacon_k_proj._is_hf_initialized = True
        if "v" in config.beacon_param:
            self.beacon_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.v_proj.bias is not None)
            self.beacon_v_proj.weight.data.zero_()
            self.beacon_v_proj._is_hf_initialized = True
        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.o_proj.bias is not None)
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

    def _init_beacon_proj(self, missing_keys):
        """Initialize the beacon projection weight with that of the ordinal projection."""
        beacon_param = self.config.beacon_param
        
        if is_deepspeed_zero3_enabled():
            # FIXME: after deepspeed initialization, some weights becomes non-zero
            # For Mistral, there are rows that are full of zeros
            # For Mistral, there are values bigger than 1e29...

            import deepspeed
            if "q" in beacon_param:
                params = [self.beacon_q_proj.weight, self.q_proj.weight]
                if self.q_proj.bias is not None:
                    params.extend([self.beacon_q_proj.bias, self.q_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_q_proj.weight.sum(-1) == 0).any() or (self.beacon_q_proj.weight > 1e29).any():
                        self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                        if self.q_proj.bias is not None:
                            self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in beacon_param:
                params = [self.beacon_k_proj.weight, self.k_proj.weight]
                if self.k_proj.bias is not None:
                    params.extend([self.beacon_k_proj.bias, self.k_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_k_proj.weight.sum(-1) == 0).any() or (self.beacon_k_proj.weight > 1e29).any():
                        self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                        if self.k_proj.bias is not None:
                            self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param:
                params = [self.beacon_v_proj.weight, self.v_proj.weight]
                if self.v_proj.bias is not None:
                    params.extend([self.beacon_v_proj.bias, self.v_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_v_proj.weight.sum(-1) == 0).any() or (self.beacon_v_proj.weight > 1e29).any():
                        self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                        if self.v_proj.bias is not None:
                            self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param:
                params = [self.beacon_o_proj.weight, self.o_proj.weight]
                if self.o_proj.bias is not None:
                    params.extend([self.beacon_o_proj.bias, self.o_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_o_proj.weight.sum(-1) == 0).any() or (self.beacon_o_proj.weight > 1e29).any():
                        self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                        if self.o_proj.bias is not None:
                            self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data
        else:
            # only copy the value in-place, without tieing the weight
            if "q" in beacon_param and any("beacon_q_proj" in missing_key for missing_key in missing_keys):
                # FIXME: some beacon weights are not initialized as zero for mistral model, why? 
                # if (self.beacon_q_proj.weight == 0).all():
                    self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                    if self.q_proj.bias is not None:
                        self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in beacon_param and any("beacon_k_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_k_proj.weight == 0).all():
                    self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                    if self.k_proj.bias is not None:
                        self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param and any("beacon_v_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_v_proj.weight == 0).all():
                    self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                    if self.v_proj.bias is not None:
                        self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param and any("beacon_o_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_o_proj.weight == 0).all():
                    self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                    if self.o_proj.bias is not None:
                        self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]

            # NOTE: there is slight redundant computation because ordinal tokens should never be projected by beacon matrices, but we are doing this for efficiency
            if "q" in self.config.beacon_param:
                ordinal_query_states = self.q_proj(hidden_states)
                beacon_query_states = self.beacon_q_proj(hidden_states)
                query_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_query_states, beacon_query_states)
                if (cur_beacon_indices == 2).any():
                    # beacon_indices == 2 means the beacon token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all beacon tokens then copy them to the replicate beacon tokens
                    query_states[:, cur_beacon_indices == 2] = beacon_query_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                query_states = self.q_proj(hidden_states)

            if "k" in self.config.beacon_param:
                ordinal_key_states = self.k_proj(hidden_states)
                beacon_key_states = self.beacon_k_proj(hidden_states)
                key_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_key_states, beacon_key_states)
                if (cur_beacon_indices == 2).any():
                    # beacon_indices == 2 means the beacon token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all beacon tokens then copy them to the replicate beacon tokens
                    key_states[:, cur_beacon_indices == 2] = beacon_key_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                key_states = self.k_proj(hidden_states)

            if "v" in self.config.beacon_param:
                ordinal_value_states = self.v_proj(hidden_states)
                beacon_value_states = self.beacon_v_proj(hidden_states)
                value_states = torch.where((cur_beacon_indices == 0)[:, None], ordinal_value_states, beacon_value_states)
                if (cur_beacon_indices == 2).any():
                    # beacon_indices == 2 means the beacon token is used to replicate the ones in previous window for parallel encoding
                    # we should slice out all beacon tokens then copy them to the replicate beacon tokens
                    value_states[:, cur_beacon_indices == 2] = beacon_value_states[:, cur_beacon_indices == 1][:, :(cur_beacon_indices == 2).sum()]
            else:
                value_states = self.v_proj(hidden_states)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states

    def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-attn_output.shape[1]:]

            if "o" in self.config.beacon_param:
                ordinal_attn_output = self.o_proj(attn_output)
                beacon_attn_output = self.beacon_o_proj(attn_output)
                attn_output = torch.where((cur_beacon_indices == 0)[:, None], ordinal_attn_output, beacon_attn_output)
            else:
                attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, beacon_size, beacon_indices = past_key_value

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralSdpaAttention(MistralAttention):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MistralModel is using MistralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        return attn_output, None, past_key_value


class MistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]

        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MistralRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, 
            key_states, 
            value_states, 
            attention_mask, 
            q_len, 
            dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MistralFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "sdpa": MistralSdpaAttention,
    "flash_attention_2": MistralFlashAttention2,
}


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window is not None and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MISTRAL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralPreTrainedModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MISTRAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # BEACON: add beacon embedding
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens._is_hf_initialized = True

        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _init_beacon_embed(self, missing_keys):
        """Initialize the beacon token embedding with that of the eos token."""
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                # deepspeed will initialize the parameters to zero
                if (self.beacon_embed_tokens.weight == 0).all():
                    if self.config.beacon_embed_init == "bos":
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    elif self.config.beacon_embed_init == "eos":
                        if isinstance(self.config.eos_token_id, list):
                            eos_token_id = self.config.eos_token_id[0]
                        else:
                            eos_token_id = self.config.eos_token_id
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                    else:
                        raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")
        else:
            if any("beacon_embed_tokens" in missing_key for missing_key in missing_keys):
                if self.config.beacon_embed_init == "bos":
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                elif self.config.beacon_embed_init == "eos":
                    if isinstance(self.config.eos_token_id, list):
                        eos_token_id = self.config.eos_token_id[0]
                    else:
                        eos_token_id = self.config.eos_token_id
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                else:
                    raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # BEACON: always use cache
        use_cache = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key, past_value, beacon_size, beacon_indices = past_key_values[0]

        # BEACON: separately embed ordinal tokens and beacon tokens because ordinal tokens do not receive gradients
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-input_ids.shape[1]:]

            ordinal_input_ids = input_ids[:, cur_beacon_indices == 0]
            beacon_input_ids = input_ids[:, cur_beacon_indices > 0]
            ordinal_inputs_embeds = self.embed_tokens(ordinal_input_ids)
            beacon_input_embeds = self.beacon_embed_tokens(beacon_input_ids - self.config.vocab_size)
            # create a new embedding tensor
            inputs_embeds = beacon_input_embeds.new_zeros(*input_ids.shape, beacon_input_embeds.shape[-1])
            inputs_embeds[:, cur_beacon_indices == 0] = ordinal_inputs_embeds
            inputs_embeds[:, cur_beacon_indices > 0] = beacon_input_embeds

        else:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        # print(f"input_ids:          {input_ids}")
        # print(f"beacon_indices:     {beacon_indices}")
        # print(f"position_ids:       {position_ids}")
        # print(f"attention_mask:\n{attention_mask == 0}")
        # x = input()
        # if x == "s":
        #     return

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # BEACON: still use tuple to organize cache
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # BEACON: slice out the past_key_value of the corresponding layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        config = model.config
        model.memory = Memory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
        )

        missing_keys = loading_info["missing_keys"]
        # NOTE: the beacon parameters may or may not be loaded from the checkpoint
        # if it is loaded from the checkpoint, we should not re-initilize it
        model.model._init_beacon_embed(missing_keys)
        # initialize weights of possible q,k,v,o,mlp
        for layer in model.model.layers:
            layer.self_attn._init_beacon_proj(missing_keys)

        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # when we directly call _native_forward, the past_key_values would be None
        if past_key_values is None:
            # NOTE: set beacon size to 0 to avoid using any beacon parameters, see Qwen2Attention.forward
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        batch_loss = None
        token_loss = None
        
        if labels is not None:
            loss, batch_loss, token_loss = compute_loss(logits, labels, shift=False)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            token_loss=token_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _beacon_forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # t1 = time.time()

        # initialize cache
        self.memory.prepare(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        # t2 = time.time()

        while not self.memory.finish:

            # t3 = time.time()

            input_ids, attention_mask, position_ids, past_key_values, labels = self.memory.step()

            # t4 = time.time()

            outputs = self._native_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

            # t5 = time.time()

            # update past_key_values
            self.memory.update_memory(outputs.past_key_values)

            # t6 = time.time()

            if labels is not None:
                # update loss
                self.memory.update_loss(outputs.batch_loss, (labels != -100).sum(-1))

            # t7 = time.time()

            # print(f"step time: {t4-t3}, forward time: {t5-t4}, update time: {t6-t5}, loss time: {t7-t6}")
            # input()

        # t8 = time.time()

        # output loss, past_key_values, and perplexity
        outputs = self.memory.output(outputs)

        # t9 = time.time()

        # print(f"output time:            {t9-t8}")
        # input()

        return outputs

    def forward(self, **kwargs):
        """Forward computation over a batch of sequences.
        """
        # only allow gradient when training
        with optional_grad_ctx(with_grad=self.training):
            # we can disable beacon to use the original mistral
            if hasattr(self, "_enable_beacon") and self._enable_beacon == False:
                return self._native_forward(**kwargs)
            else:
                return self._beacon_forward(**kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
