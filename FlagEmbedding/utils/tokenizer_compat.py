"""Tokenizer compatibility helpers for supported Transformers versions."""

from collections.abc import Mapping
from typing import Any, List, Optional, Sequence


def pad_with_compat(tokenizer: Any, encoded_inputs: Any, **kwargs: Any) -> Any:
    """Pad tokenized examples across Transformers v4 and v5.

    Several FlagEmbedding inference paths sort individually tokenized
    examples before padding, producing a list of mappings.  Transformers
    releases do not all handle that representation consistently, so normalize
    it to the equivalent mapping-of-lists representation before calling
    ``tokenizer.pad``.

    Inputs that are already mappings, empty inputs, and other supported
    tokenizer inputs are passed through unchanged.
    """
    if (
        isinstance(encoded_inputs, (list, tuple))
        and encoded_inputs
        and isinstance(encoded_inputs[0], Mapping)
    ):
        encoded_inputs = {
            key: [example[key] for example in encoded_inputs]
            for key in encoded_inputs[0].keys()
        }

    return tokenizer.pad(encoded_inputs, **kwargs)


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode token ids without dropping unknown or other special tokens."""
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _truncate_second_sequence(
    tokenizer: Any,
    first_ids: Sequence[int],
    second_ids: Sequence[int],
    max_length: Optional[int],
    truncation: Any,
) -> List[int]:
    """Apply the old ``only_second`` truncation behavior to token ids."""
    second_ids = list(second_ids)
    truncation = getattr(truncation, "value", truncation)
    if max_length is None or truncation not in (True, "only_second"):
        return second_ids

    tokens_to_remove = len(first_ids) + len(second_ids) - max_length
    if tokens_to_remove <= 0 or len(second_ids) <= tokens_to_remove:
        # This matches the legacy tokenizer behavior: only the second sequence
        # may be truncated, so an oversized first sequence is left untouched.
        return second_ids

    if getattr(tokenizer, "truncation_side", "right") == "left":
        return second_ids[tokens_to_remove:]
    return second_ids[:-tokens_to_remove]


def prepare_for_model_compat(
    tokenizer: Any,
    first_ids: Sequence[int],
    second_ids: Optional[Sequence[int]] = None,
    *,
    truncation: Any = None,
    max_length: Optional[int] = None,
    padding: Any = False,
    return_attention_mask: Optional[bool] = None,
    return_token_type_ids: Optional[bool] = None,
    add_special_tokens: bool = True,
    **kwargs: Any,
) -> dict:
    """Prepare a pair of tokenized sequences across Transformers v4 and v5.

    Transformers v5 removed the id-level ``prepare_for_model`` API from
    tokenizers. For encoder-only rerankers, the supported replacement is the
    tokenizer call with a text pair. The ids are decoded with special tokens
    preserved so ``unk_token_id`` and model-specific special tokens are not
    silently lost before the pair is tokenized again.

    Decoder-only rerankers use this helper with ``add_special_tokens=False``.
    In that mode no text round-trip is needed; the second sequence is truncated
    directly while respecting ``tokenizer.truncation_side``.
    """
    has_pair = second_ids is not None
    if second_ids is None:
        second_ids = []

    legacy_prepare_for_model = getattr(tokenizer, "prepare_for_model", None)
    if callable(legacy_prepare_for_model):
        return legacy_prepare_for_model(
            list(first_ids),
            list(second_ids) if has_pair else None,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    if add_special_tokens:
        tokenizer_kwargs = dict(
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            add_special_tokens=True,
        )
        tokenizer_kwargs.update(kwargs)
        return tokenizer(
            _decode_token_ids(tokenizer, first_ids),
            _decode_token_ids(tokenizer, second_ids) if has_pair else None,
            **tokenizer_kwargs,
        )

    first_ids = list(first_ids)
    second_ids = _truncate_second_sequence(
        tokenizer,
        first_ids,
        second_ids,
        max_length,
        truncation,
    )
    input_ids = first_ids + second_ids
    result = {"input_ids": input_ids}

    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in getattr(tokenizer, "model_input_names", [])
    if return_token_type_ids is None:
        return_token_type_ids = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
    if return_attention_mask:
        result["attention_mask"] = [1] * len(input_ids)
    if return_token_type_ids:
        result["token_type_ids"] = [0] * len(first_ids) + [0] * len(second_ids)

    return result
