"""Shared helpers for single-device inference batch-size probing.

Several inference paths probe for the largest workable batch size: they run a
trial forward pass and, on out-of-memory (OOM) errors, shrink the batch size
and retry. The helpers here implement the policy for that loop:

* Only genuine OOM errors trigger a retry; any other ``RuntimeError``
  propagates immediately with its original traceback.
* Retrying stops (re-raising the OOM error) once the batch size would drop
  below 1 -- a single sample cannot fit, so further shrinking is pointless.
* The batch size therefore never drops below 1 inside the probing loop.
"""

from typing import Union


def is_oom_error(error: BaseException) -> bool:
    """Return True if *error* is an out-of-memory failure.

    Matches ``torch.cuda.OutOfMemoryError`` (a ``RuntimeError`` subclass) via
    its message, as well as allocator messages such as "CUDA out of memory"
    or the CPU allocator's "not enough memory" errors.
    """
    message = str(error).lower()
    return "out of memory" in message or "not enough memory" in message


def shrink_batch_size_on_oom(batch_size: int, error: RuntimeError) -> int:
    """Return the batch size to retry with after an OOM *error*.

    Re-raises the original *error* when the reduced batch size would drop
    below 1: a single sample does not fit, so retrying is pointless.
    """
    new_batch_size = batch_size * 3 // 4
    if new_batch_size < 1:
        raise error
    return new_batch_size


def validate_batch_size(batch_size: Union[int, None]) -> None:
    """Reject non-positive caller-supplied batch sizes early.

    ``None`` is allowed (callers fall back to the configured default).
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
