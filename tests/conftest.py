"""
Common pytest fixtures and configuration for FlagEmbedding tests.
"""

import os
import pytest
import torch
from packaging import version
import transformers

# Check if we're using transformers v5+
TF_VER = version.parse(getattr(transformers, "__version__", "0.0.0"))
IS_TF_V5_OR_HIGHER = TF_VER >= version.parse("5.0.0")


@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def transformers_version():
    """Return the transformers version."""
    return TF_VER
