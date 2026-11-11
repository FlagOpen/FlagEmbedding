import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import EncoderOnlyEmbedderM3Model


class DummyM3Model(EncoderOnlyEmbedderM3Model):
    def __init__(self, temperature=0.05):
        self.temperature = temperature


def test_m3_pairwise_scoring_matches_all_pairs():
    torch.manual_seed(42)
    B = 5
    L_q = 8
    L_p = 12
    dim = 64
    temp = 0.05

    model = DummyM3Model(temperature=temp)

    q_dense = torch.randn(B, dim)
    p_dense = torch.randn(B, dim)
    q_sparse = torch.randn(B, dim)
    p_sparse = torch.randn(B, dim)
    q_colbert = torch.randn(B, L_q, dim)
    p_colbert = torch.randn(B, L_p, dim)
    q_mask = torch.ones(B, L_q)

    # 1. Dense score
    all_dense = model.compute_dense_score(q_dense, p_dense)
    inx = torch.arange(B)
    expected_dense = all_dense[inx, inx]
    actual_dense = model.compute_dense_score_pairwise(q_dense, p_dense)
    assert torch.allclose(expected_dense, actual_dense, atol=1e-5)

    # 2. Sparse score
    all_sparse = model.compute_sparse_score(q_sparse, p_sparse)
    expected_sparse = all_sparse[inx, inx]
    actual_sparse = model.compute_sparse_score_pairwise(q_sparse, p_sparse)
    assert torch.allclose(expected_sparse, actual_sparse, atol=1e-5)

    # 3. ColBERT score
    all_colbert = model.compute_colbert_score(q_colbert, p_colbert, q_mask=q_mask)
    expected_colbert = all_colbert[inx, inx]
    actual_colbert = model.compute_colbert_score_pairwise(q_colbert, p_colbert, q_mask=q_mask)
    assert torch.allclose(expected_colbert, actual_colbert, atol=1e-5)
