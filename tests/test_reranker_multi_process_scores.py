"""Worker-score aggregation must preserve the single-device output layout."""

import math
from queue import Queue

import numpy as np
import pytest

from FlagEmbedding.abc.inference import AbsReranker


class AggregationReranker(AbsReranker):
    """Exercise aggregation with precomputed worker results, without model downloads."""

    def compute_score_single_gpu(self, sentence_pairs, **kwargs):
        raise AssertionError(
            "The aggregation test must use the supplied worker results"
        )


@pytest.mark.parametrize("pair_count", [1, 3, 4, 7])
@pytest.mark.parametrize("layer_count", [1, 2, 3])
def test_worker_scores_preserve_layer_and_pair_axes(pair_count, layer_count):
    reranker = AggregationReranker("unused", devices=["cpu", "cpu"])
    sentence_pairs = [(f"query {i}", f"passage {i}") for i in range(pair_count)]
    expected = np.arange(layer_count * pair_count, dtype=float).reshape(
        layer_count, pair_count
    )
    # Both layerwise and lightweight rerankers collapse a single score layer.
    if layer_count == 1:
        expected = expected[0]

    chunk_size = math.ceil(pair_count / len(reranker.target_devices))
    chunks = [
        (chunk_id, start, min(start + chunk_size, pair_count))
        for chunk_id, start in enumerate(range(0, pair_count, chunk_size))
    ]
    input_queue, output_queue = Queue(), Queue()
    # Complete out of order, as real workers may. Their scores are Python lists.
    for chunk_id, start, end in reversed(chunks):
        output_queue.put([chunk_id, expected[..., start:end].tolist()])
    pool = {
        "input": input_queue,
        "output": output_queue,
        "processes": [None] * len(reranker.target_devices),
    }

    scores = reranker.encode_multi_process(sentence_pairs, pool, normalize=True)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == expected.shape
    np.testing.assert_array_equal(scores, expected)
    assert output_queue.empty()
    for chunk_id, start, end in chunks:
        assert input_queue.get_nowait() == [
            chunk_id,
            sentence_pairs[start:end],
            {"normalize": True},
        ]
    assert input_queue.empty()
