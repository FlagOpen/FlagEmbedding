"""Regression test for #907.

`device=` (singular) must be honored as an alias for `devices=` instead of being
silently swallowed by **kwargs — which left `devices=None`, so the model spanned ALL
visible GPUs and OOM'd. Model-free: only exercises argument handling in the abstract
base classes (no weights downloaded, no GPU required).
"""
import pytest

from FlagEmbedding.abc.inference.AbsEmbedder import AbsEmbedder
from FlagEmbedding.abc.inference.AbsReranker import AbsReranker


class _DummyEmbedder(AbsEmbedder):
    def encode_single_device(self, *args, **kwargs):  # abstract stub
        pass


class _DummyReranker(AbsReranker):
    def compute_score_single_gpu(self, *args, **kwargs):  # abstract stub
        pass


@pytest.mark.parametrize("cls", [_DummyEmbedder, _DummyReranker])
def test_singular_device_aliases_to_devices(cls):
    # `device=` should be honored, not dropped into **kwargs.
    obj = cls("dummy-model-name", device="cuda:5")
    assert obj.target_devices == ["cuda:5"]
    # ...and must NOT leak as a stray attribute or into self.kwargs.
    assert not hasattr(obj, "device")
    assert "device" not in obj.kwargs


@pytest.mark.parametrize("cls", [_DummyEmbedder, _DummyReranker])
def test_plural_devices_still_works(cls):
    obj = cls("dummy-model-name", devices="cuda:3")
    assert obj.target_devices == ["cuda:3"]


@pytest.mark.parametrize("cls", [_DummyEmbedder, _DummyReranker])
def test_explicit_devices_wins_over_device(cls):
    obj = cls("dummy-model-name", devices="cuda:2", device="cuda:9")
    assert obj.target_devices == ["cuda:2"]
