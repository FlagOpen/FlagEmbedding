"""
Test basic functionality of BGE embedder models with Transformers v5.

This test loads a small/public BGE checkpoint and runs a single encode on toy strings,
verifying that the shape/dtype are correct and that cosine similarity is sane.
"""
import pytest
import torch
import numpy as np
from FlagEmbedding import FlagModel

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_bge_embedder_basic(device):
    """Test basic functionality of BGE embedder."""
    # Load a small BGE model
    model_name = "BAAI/bge-base-en-v1.5"
    model = FlagModel(model_name, device=device)
    
    # Test encoding single strings
    query = "What is the capital of France?"
    passage = "Paris is the capital and most populous city of France."
    
    # Get embeddings
    query_embedding = model.encode(query)
    passage_embedding = model.encode(passage)
    
    # Check shapes and types
    assert isinstance(query_embedding, np.ndarray)
    assert isinstance(passage_embedding, np.ndarray)
    assert query_embedding.ndim == 1  # Should be a 1D vector
    assert passage_embedding.ndim == 1  # Should be a 1D vector
    
    # Check that embeddings have reasonable values
    assert not np.isnan(query_embedding).any()
    assert not np.isnan(passage_embedding).any()
    
    # Check cosine similarity is reasonable (should be high for related texts)
    similarity = cosine_similarity(query_embedding, passage_embedding)
    assert 0 <= similarity <= 1  # Cosine similarity range
    assert similarity > 0.5  # These texts should be somewhat similar

def test_bge_embedder_batch(device):
    """Test batch encoding with BGE embedder."""
    # Load a small BGE model
    model_name = "BAAI/bge-base-en-v1.5"
    model = FlagModel(model_name, device=device)
    
    # Test batch encoding
    queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ]
    
    # Get embeddings
    embeddings = model.encode(queries)
    
    # Check shapes and types
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2  # Should be a 2D array (batch_size x embedding_dim)
    assert embeddings.shape[0] == len(queries)
    
    # Check that embeddings have reasonable values
    assert not np.isnan(embeddings).any()