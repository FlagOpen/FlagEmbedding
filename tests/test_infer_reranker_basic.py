"""
Test basic functionality of reranker models with Transformers v5.

This test instantiates a lightweight reranker and calls compute_score on query/doc pairs
to validate the forward pass.
"""

import pytest
import torch
import numpy as np
from FlagEmbedding import FlagReranker


def test_reranker_basic(device):
    """Test basic functionality of reranker."""
    # Load a lightweight reranker model
    model_name = "BAAI/bge-reranker-base"
    model = FlagReranker(model_name, device=device)

    # Test scoring a single query-document pair
    query = "What is the capital of France?"
    passage = "Paris is the capital and most populous city of France."

    # Get score
    pair = [(query, passage)]
    scores = model.compute_score(pair)
    score = scores[0]

    # Check score type and range
    assert isinstance(score, float)
    # Scores are typically in a reasonable range (model-dependent)
    assert -100 < score < 100


def test_reranker_batch(device):
    """Test batch scoring with reranker."""
    # Load a lightweight reranker model
    model_name = "BAAI/bge-reranker-base"
    model = FlagReranker(model_name, device=device)

    # Test batch scoring
    query = "What is the capital of France?"
    passages = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital and largest city of Germany.",
        "London is the capital and largest city of England and the United Kingdom.",
    ]

    # Create pairs for scoring
    pairs = [(query, passage) for passage in passages]

    # Get scores
    scores = model.compute_score(pairs)

    # Check scores shape and type
    assert isinstance(scores, list)
    assert len(scores) == len(passages)
    assert all(isinstance(score, float) for score in scores)

    # Check that Paris (correct answer) gets highest score
    paris_score = scores[0]
    assert paris_score == max(scores), "Paris should have the highest score"
