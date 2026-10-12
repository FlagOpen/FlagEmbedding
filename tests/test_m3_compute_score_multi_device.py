"""
Test that BGE-M3 compute_score handles a single (query, passage) pair on multiple devices.

The multi-process pool is stubbed, so no model is loaded and no processes are started.
"""
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder

MODES = ["colbert", "sparse", "dense", "sparse+dense", "colbert+sparse+dense"]


def _make_embedder(received):
    embedder = M3Embedder.__new__(M3Embedder)
    embedder.target_devices = ["cpu", "cpu"]
    embedder.batch_size = 256
    embedder.query_max_length = 512
    embedder.passage_max_length = 512
    embedder.start_multi_process_pool = lambda worker: {"processes": [None, None]}
    embedder.stop_multi_process_pool = lambda pool: None

    def compute_score_multi_process(sentence_pairs, pool, **kwargs):
        received.append(sentence_pairs)
        return {mode: [0.5] * len(sentence_pairs) for mode in MODES}

    embedder.compute_score_multi_process = compute_score_multi_process
    return embedder


def test_m3_compute_score_single_pair_multi_device():
    received = []
    embedder = _make_embedder(received)

    pair = ("what is BGE M3?", "BGE M3 is an embedding model.")
    scores = embedder.compute_score(pair)

    # The pair is scored as one pair, not as two separate strings
    assert received == [[pair]]
    # Same output shape as a single device: one float per mode
    assert scores == {mode: 0.5 for mode in MODES}


def test_m3_compute_score_pair_list_multi_device():
    received = []
    embedder = _make_embedder(received)

    pairs = [("q1", "p1"), ("q2", "p2")]
    scores = embedder.compute_score(pairs)

    assert received == [pairs]
    assert scores == {mode: [0.5, 0.5] for mode in MODES}
