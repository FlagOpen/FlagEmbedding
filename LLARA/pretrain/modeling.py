import sys
from typing import Optional, List, Union, Tuple

import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.idefics.modeling_idefics import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaModel
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, logging
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import torch.distributed as dist

logger = logging.get_logger(__name__)

class NewLlamaModel(LlamaModel):
    add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config._attn_implementation == "flash_attention_2":
            raise ValueError(
                "You can not use flash attention to pretrain"
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        summarize_suffix_ids = [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871,
                                32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]
        predict_suffix_ids = [9162, 8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000,
                              32009, 32012, 32001, 32010, 32003, 32006, 32015]

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            for i in range(len(position_ids)):
                position_ids[i][-len(predict_suffix_ids):] = copy.deepcopy(position_ids[i][
                                                             -len(summarize_suffix_ids) - len(predict_suffix_ids): -len(
                                                                 summarize_suffix_ids)])

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        causal_mask[:,
                :,
                -len(predict_suffix_ids) :,
                -len(predict_suffix_ids) - len(summarize_suffix_ids): -len(predict_suffix_ids),
        ] = torch.finfo(inputs_embeds.dtype).min

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #             attention_mask,
        #             inputs_embeds=input_tensor,
        #             past_key_values_length=past_seen_tokens,
        #             is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class PreLlamaModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = NewLlamaModel(config)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        """
        prompt type1: "{}", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>
        token ids: [376, ..., 9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871, 
                    32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]

        prompt type2: "{}", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>
        token ids: [376, ..., 8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000, 32009, 
                    32012, 32001, 32010, 32003, 32006, 32015]

        [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901,
        29871, 32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014,
        8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000,
        32009, 32012, 32001, 32010, 32003, 32006, 32015]

        Maybe only one of them will appear, or both may appear. We consider all possibilities here.
        """

        self.summarize_prompt_ids = [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901, 29871,
                                     32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014]
        self.predict_prompt_ids = [9162, 8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 32000,
                                   32009, 32012, 32001, 32010, 32003, 32006, 32015]

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_summarize_ids: Optional[torch.LongTensor] = None,
            output_predict_ids: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        ar_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ar_loss = loss_fct(shift_logits, shift_labels)

        """
        prompt: ", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>", predict the following passage within eight words: <s9><s10><s11><s12><s13><s14><s15><s16>
        token ids: [9162, 19138, 675, 278, 2038, 13382, 2629, 9475, 3838, 29901,
                    29871, 32008, 32011, 32004, 32013, 32007, 32005, 32002, 32014,
                    9162, 8500, 278, 1494, 13382, 2629, 9475, 3838, 29901, 29871, 
                    32000, 32009, 32012, 32001, 32010, 32003, 32006, 32015]
        token ids[-26:-18] —— <s1><s2><s3><s4><s5><s6><s7><s8>
        token ids[-8:] —— <s9><s10><s11><s12><s13><s14><s15><s16>
        """
        bow_summarize_loss = 0
        if output_summarize_ids is not None:
            special_logits = logits[:, -len(self.predict_prompt_ids) - 8:-len(self.predict_prompt_ids), :]
            special_logits, _ = torch.max(special_logits, dim=1)
            bow_summarize_loss = 0
            possibility = self.log_softmax(special_logits)
            batch_num = 0
            for p, temp_output_ids in zip(possibility, output_summarize_ids):
                unique_useful_ids = torch.unique(temp_output_ids[temp_output_ids > 2])
                if len(unique_useful_ids) > 0:
                    bow_summarize_loss -= torch.mean(p[unique_useful_ids])
                    batch_num += 1
            if batch_num > 0:
                bow_summarize_loss /= batch_num
                bow_summarize_loss /= 10

        bow_predict_loss = 0
        if output_predict_ids is not None:
            special_logits = logits[:, -8:, :]
            special_logits, _ = torch.max(special_logits, dim=1)
            bow_predict_loss = 0
            possibility = self.log_softmax(special_logits)
            batch_num = 0
            for p, temp_output_ids in zip(possibility, output_predict_ids):
                unique_useful_ids = torch.unique(temp_output_ids[temp_output_ids > 2])
                if len(unique_useful_ids) > 0:
                    bow_predict_loss -= torch.mean(p[unique_useful_ids])
                    batch_num += 1
            if batch_num > 0:
                bow_predict_loss /= batch_num
                bow_predict_loss /= 10

        if bow_summarize_loss > 0 and bow_predict_loss > 0:
            bow_loss = (bow_summarize_loss + bow_predict_loss) / 2
        elif bow_summarize_loss > 0:
            bow_loss = bow_summarize_loss
        elif bow_predict_loss > 0:
            bow_loss = bow_predict_loss
        else:
            bow_loss = None

        if ar_loss is not None and bow_loss is not None:
            loss = ar_loss + bow_loss
        elif ar_loss is None:
            loss = bow_loss
        else:
            loss = ar_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PreModel(nn.Module):
    def __init__(self,
                 model: AutoModel = None,
                 ):
        super().__init__()
        self.model = model

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)