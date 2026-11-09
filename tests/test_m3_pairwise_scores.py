import pytest
import torch

from FlagEmbedding.finetune.embedder.encoder_only.m3.modeling import EncoderOnlyEmbedderM3Model
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder


class Features(dict):
    def to(self, device):
        return self


def tokenize(texts, **kwargs):
    lengths = [int(text.split(":")[1]) for text in texts]
    mask = torch.arange(max(lengths) + 1) <= torch.tensor(lengths)[:, None]
    return Features(attention_mask=mask, texts=texts)


class FixedEncoder(torch.nn.Module):
    temperature = 0.7
    _compute_similarity = EncoderOnlyEmbedderM3Model._compute_similarity
    compute_dense_score = EncoderOnlyEmbedderM3Model.compute_dense_score
    compute_sparse_score = EncoderOnlyEmbedderM3Model.compute_sparse_score
    compute_colbert_score = EncoderOnlyEmbedderM3Model.compute_colbert_score

    def forward(self, features, **kwargs):
        texts = features["texts"]
        ids = torch.tensor([int(text.split(":")[2]) for text in texts], dtype=torch.float32)
        sign = 1 if texts[0].startswith("q:") else -1
        mask = features["attention_mask"][:, 1:]
        return {
            "dense_vecs": torch.stack((ids, ids + sign), dim=-1),
            "sparse_vecs": torch.stack((ids, sign * ids, ids + 1), dim=-1),
            "colbert_vecs": (sign * mask.float()).unsqueeze(-1),
        }


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("weights", [None, [0.4, 0.2, 0.4]])
def test_compute_score_matches_upstream_diagonal(single, weights):
    pairs = [
        ("q:2:2", "p:513:4"),
        ("q:4:3", "p:512:7"),
        ("q:1:5", "p:2:1"),
    ]
    if single:
        pairs = pairs[:1]

    model = FixedEncoder()
    embedder = object.__new__(M3Embedder)
    embedder.model = model
    embedder.tokenizer = tokenize
    embedder.target_devices = ["cpu"]
    embedder.pool = None
    actual = embedder.compute_score(
        pairs[0] if single else pairs, batch_size=2, max_query_length=8,
        max_passage_length=1024, weights_for_different_modes=weights,
    )

    expected_batches = []
    dense_weight, sparse_weight, colbert_weight = weights or [1.0, 1.0, 1.0]
    for start in range(0, len(pairs), 2):
        queries, passages = zip(*pairs[start:start + 2])
        q_features = tokenize(queries)
        q, p = model(q_features), model(tokenize(passages))
        dense = model.compute_dense_score(q["dense_vecs"], p["dense_vecs"]).diagonal()
        sparse = model.compute_sparse_score(q["sparse_vecs"], p["sparse_vecs"]).diagonal()
        colbert = model.compute_colbert_score(
            q["colbert_vecs"], p["colbert_vecs"], q_mask=q_features["attention_mask"],
        ).diagonal()
        weighted_dense_sparse = dense * dense_weight + sparse * sparse_weight
        expected_batches.append({
            "dense": dense,
            "sparse": sparse,
            "colbert": colbert,
            "sparse+dense": weighted_dense_sparse / (dense_weight + sparse_weight),
            "colbert+sparse+dense": (
                weighted_dense_sparse + colbert * colbert_weight
            ) / (dense_weight + sparse_weight + colbert_weight),
        })

    assert actual.keys() == expected_batches[0].keys()
    for mode, scores in actual.items():
        assert isinstance(scores, float if single else list)
        expected = torch.cat([batch[mode] for batch in expected_batches])
        torch.testing.assert_close(torch.as_tensor(scores).reshape(-1), expected)
