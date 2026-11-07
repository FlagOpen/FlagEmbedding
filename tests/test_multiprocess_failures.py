import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np
import pytest
import torch

from FlagEmbedding import FlagModel
from FlagEmbedding.abc.inference import AbsEmbedder, AbsReranker
from FlagEmbedding.inference.embedder.decoder_only.icl import ICLLLMEmbedder
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder


_STARTED_POOLS = []


class TrackPools:
    def start_multi_process_pool(self, *args):
        pool = super().start_multi_process_pool(*args)
        _STARTED_POOLS.append((pool, [p.pid for p in pool["processes"]]))
        return pool


class ToyEmbedder(TrackPools, AbsEmbedder):
    def __init__(self):
        super().__init__("local", devices=["cpu", "cpu"], use_fp16=False)
        self.model = torch.nn.Linear(1, 1)

    def encode_single_device(self, sentences, **kwargs):
        if "fail" in sentences:
            raise ValueError("controlled inference failure")
        if "exit" in sentences:
            os._exit(7)
        # Force the later chunk to finish first.
        if sentences[0] == "slow":
            time.sleep(0.2)
        return np.array([[len(sentence)] for sentence in sentences], dtype=np.float32)


class ToyReranker(TrackPools, AbsReranker):
    def __init__(self):
        super().__init__("local", devices=["cpu", "cpu"], use_fp16=False)
        self.model = torch.nn.Linear(1, 1)

    def compute_score_single_gpu(self, pairs, **kwargs):
        return ToyEmbedder.encode_single_device(self, [pair[0] for pair in pairs]).reshape(-1)


class ToyICL(TrackPools, ICLLLMEmbedder):
    def __init__(self):
        AbsEmbedder.__init__(self, "local", devices=["cpu", "cpu"], use_fp16=False)
        self.model = torch.nn.Linear(1, 1)
        self.query_pool = None

    encode_queries_single_device = ToyEmbedder.encode_single_device


class ToyM3(TrackPools, M3Embedder):
    def __init__(self):
        AbsEmbedder.__init__(self, "local", devices=["cpu", "cpu"], use_fp16=False)
        self.model = torch.nn.Linear(1, 1)

    def compute_score_single_device(self, pairs, **kwargs):
        scores = ToyEmbedder.encode_single_device(self, [pair[0] for pair in pairs]).reshape(-1).tolist()
        return {mode: scores for mode in (
            "dense", "sparse", "colbert", "sparse+dense", "colbert+sparse+dense"
        )}


class LocalFlagModel(TrackPools, FlagModel):
    pass


@pytest.fixture
def started_pools():
    _STARTED_POOLS.clear()
    yield _STARTED_POOLS
    for pool, _ in _STARTED_POOLS:
        try:
            AbsEmbedder.stop_multi_process_pool(pool)
        except ValueError:
            # A failed request or M3's per-call pool may already have closed it.
            pass


def invoke(call, started_pools):
    """Bound a broken queue wait, including on the unfixed source."""
    outcome = queue.Queue()

    def run():
        try:
            outcome.put((True, call()))
        except BaseException as error:
            outcome.put((False, error))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(60)
    if thread.is_alive():
        # Release the old collector before failing, so the regression itself leaks no
        # blocked threads. Production queues and spawned workers are otherwise real.
        for pool, _ in started_pools:
            try:
                for chunk in range(2):
                    pool["output"].put([chunk, np.zeros((1, 1))])
            except ValueError:
                pass
        thread.join(5)
        pytest.fail("The caller blocked after an inference worker failed")
    success, value = outcome.get_nowait()
    if not success:
        raise value
    return value


@pytest.mark.parametrize("failure, message", [
    ("fail", "ValueError: controlled inference failure"),
    ("exit", "worker"),
])
def test_encode_worker_failure_and_recovery(started_pools, failure, message):
    model = ToyEmbedder()
    try:
        with pytest.raises(RuntimeError, match=message):
            invoke(lambda: model.encode([failure, "ok"]), started_pools)
        assert model.pool is None
        failed_pids = started_pools[0][1]
        assert not set(failed_pids) & {p.pid for p in mp.active_children()}

        actual = invoke(lambda: model.encode(["slow", "a", "abc"]), started_pools)
        np.testing.assert_array_equal(actual, [[4], [1], [3]])
        assert len(started_pools) == 2
    finally:
        model.stop_self_pool()


@pytest.mark.parametrize("kind", ["reranker", "icl", "m3"])
def test_specialized_worker_failure_and_recovery(started_pools, kind):
    factories = {"reranker": ToyReranker, "icl": ToyICL, "m3": ToyM3}
    model = factories[kind]()

    def call(texts):
        if kind == "icl":
            return model.encode_queries(texts)
        return model.compute_score([(text, "passage") for text in texts])

    try:
        with pytest.raises(RuntimeError, match="ValueError: controlled inference failure"):
            invoke(lambda: call(["fail", "ok"]), started_pools)
        assert model.pool is None
        if kind == "icl":
            assert model.query_pool is None
        assert not set(started_pools[0][1]) & {p.pid for p in mp.active_children()}

        actual = invoke(lambda: call(["slow", "a", "abc"]), started_pools)
        if kind == "m3":
            assert all(scores == [4, 1, 3] for scores in actual.values())
        else:
            np.testing.assert_array_equal(actual, [[4], [1], [3]] if kind == "icl" else [4, 1, 3])
        assert len(started_pools) == 2
    finally:
        model.stop_self_pool()
        if kind == "icl":
            model.stop_self_query_pool()


def test_real_model_forward_failure_and_recovery(tmp_path, started_pools):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    vocabulary = {token: i for i, token in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "d"]
    )}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", model_max_length=64,
        model_input_names=["input_ids", "token_type_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(0)
    backbone = XLMRobertaModel(XLMRobertaConfig(
        vocab_size=len(vocabulary), hidden_size=8, intermediate_size=16,
        num_attention_heads=2, num_hidden_layers=1, max_position_embeddings=8,
        pad_token_id=0,
    ))
    backbone.save_pretrained(tmp_path)

    model = LocalFlagModel(str(tmp_path), devices=["cpu", "cpu"], use_fp16=False)
    too_long = "a b c d a b c d a b"
    valid = ["a", "b c", "d"]
    # A genuine embedding-layer IndexError, not an injected inference exception.
    with pytest.raises(IndexError):
        model.encode_single_device([too_long], device="cpu", max_length=64)
    expected = model.encode_single_device(valid, device="cpu", max_length=8)
    try:
        with pytest.raises(RuntimeError, match="IndexError"):
            invoke(lambda: model.encode([too_long, "a"], max_length=64), started_pools)
        assert model.pool is None
        actual = invoke(lambda: model.encode(valid, max_length=8), started_pools)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        assert not set(started_pools[0][1]) & {p.pid for p in mp.active_children()}
    finally:
        model.stop_self_pool()

