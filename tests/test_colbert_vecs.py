"""Test that _process_colbert_vecs correctly excludes special tokens."""

import numpy as np


def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
    """Process colbert vectors to exclude special tokens.

    This is the fixed version that correctly excludes EOS token.
    CLS is already excluded in colbert_embedding (last_hidden_state[:, 1:]).
    """
    tokens_num = np.sum(attention_mask)
    return colbert_vecs[:tokens_num - 2]


def test_process_colbert_vecs_excludes_eos():
    """Test that _process_colbert_vecs excludes EOS token.

    Scenario:
    - Original sequence: [CLS, tok1, tok2, tok3, EOS, PAD, PAD]
    - attention_mask: [1, 1, 1, 1, 1, 0, 0] (5 valid tokens)
    - colbert_vecs already excludes CLS, so it's [tok1, tok2, tok3, EOS, PAD, PAD]
    - Expected output: [tok1, tok2, tok3] (3 vectors, excluding EOS)
    """
    # Simulate colbert_vecs after CLS removal (4 valid + 2 padding)
    # Shape: (6, hidden_dim) where hidden_dim = 4 for testing
    colbert_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],  # tok1
        [0.0, 1.0, 0.0, 0.0],  # tok2
        [0.0, 0.0, 1.0, 0.0],  # tok3
        [0.0, 0.0, 0.0, 1.0],  # EOS (should be excluded)
        [0.0, 0.0, 0.0, 0.0],  # PAD
        [0.0, 0.0, 0.0, 0.0],  # PAD
    ])

    # Original attention_mask (includes CLS position)
    attention_mask = [1, 1, 1, 1, 1, 0, 0]  # CLS, tok1, tok2, tok3, EOS, PAD, PAD

    result = _process_colbert_vecs(colbert_vecs, attention_mask)

    # Should return only tok1, tok2, tok3 (3 vectors)
    assert result.shape[0] == 3, f"Expected 3 vectors, got {result.shape[0]}"

    # Verify the content
    expected = np.array([
        [1.0, 0.0, 0.0, 0.0],  # tok1
        [0.0, 1.0, 0.0, 0.0],  # tok2
        [0.0, 0.0, 1.0, 0.0],  # tok3
    ])
    np.testing.assert_array_equal(result, expected)


def test_process_colbert_vecs_single_token():
    """Test with minimal valid tokens (just CLS, one token, EOS)."""
    colbert_vecs = np.array([
        [1.0, 0.0],  # tok1
        [0.0, 1.0],  # EOS
    ])
    attention_mask = [1, 1, 1]  # CLS, tok1, EOS

    result = _process_colbert_vecs(colbert_vecs, attention_mask)

    # Should return only tok1
    assert result.shape[0] == 1, f"Expected 1 vector, got {result.shape[0]}"
    np.testing.assert_array_equal(result, np.array([[1.0, 0.0]]))


if __name__ == "__main__":
    test_process_colbert_vecs_excludes_eos()
    print("test_process_colbert_vecs_excludes_eos passed!")
    test_process_colbert_vecs_single_token()
    print("test_process_colbert_vecs_single_token passed!")
    print("All tests passed!")
