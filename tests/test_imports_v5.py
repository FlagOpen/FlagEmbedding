"""
Test that imports work with Transformers v5.

This test verifies that the compatibility layer in FlagEmbedding/utils/transformers_compat.py
properly handles the the removal of is_torch_fx_available in Transformers v5
"""

import pytest
import transformers
from packaging import version

# Import the compatibility layer
from FlagEmbedding.utils.transformers_compat import is_torch_fx_available

# Check if we're using transformers v5+
TF_VER = version.parse(getattr(transformers, "__version__", "0.0.0"))
IS_TF_V5_OR_HIGHER = TF_VER >= version.parse("5.0.0")


# Import the files mentioned in issue #1561 that use is_torch_fx_available
def test_import_modeling_minicpm_reranker_inference():
    """Test importing the modeling_minicpm_reranker module from inference."""
    from FlagEmbedding.inference.reranker.decoder_only.models.modeling_minicpm_reranker import (
        LayerWiseMiniCPMForCausalLM,
    )

    assert LayerWiseMiniCPMForCausalLM is not None


def test_import_modeling_minicpm_reranker_finetune():
    """Test importing the modeling_minicpm_reranker module from finetune."""
    from FlagEmbedding.finetune.reranker.decoder_only.layerwise.modeling_minicpm_reranker import (
        LayerWiseMiniCPMForCausalLM,
    )

    assert LayerWiseMiniCPMForCausalLM is not None


@pytest.mark.skipif(not IS_TF_V5_OR_HIGHER, reason="Only relevant for Transformers v5+")
def test_is_torch_fx_available_v5():
    """Test that is_torch_fx_available works with Transformers v5."""
    # This should not raise an exception
    result = is_torch_fx_available()
    # The result depends on whether torch.fx is available, but the function should work
    assert isinstance(result, bool)


def test_transformers_version(transformers_version):
    """Test that we can detect the transformers version."""
    assert transformers_version is not None
    print(f"Transformers version: {transformers_version}")
