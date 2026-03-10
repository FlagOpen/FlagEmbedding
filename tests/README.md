# FlagEmbedding Tests

This directory contains tests for the FlagEmbedding library, including compatibility tests for Transformers 5.0.

## Test Files

- `test_imports_v5.py`: Tests that imports work with Transformers v5, particularly the compatibility layer for `is_torch_fx_available`.
- `test_infer_embedder_basic.py`: Tests basic functionality of BGE embedder models with a small public checkpoint.
- `test_infer_reranker_basic.py`: Tests basic functionality of reranker models.

## Running Tests

1. create a python venv `python -m venv pytest_venv`
2. activate venv  `source pytest_venv/bin/activate`
3. install pytest `pip install pytest`
4. install flagembedding package in development mode: `pip install -e .`

Then run the tests using pytest:

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_imports_v5.py

# Run with verbose output
pytest -v tests/
```

## Transformers 5.0 Compatibility

The tests verify that FlagEmbedding works with Transformers 5.0, which removed the `is_torch_fx_available` function.
The compatibility layer in `FlagEmbedding/utils/transformers_compat.py` provides this function for backward compatibility.

**Note:** Transformers 5.0 requires Python 3.10 or higher. If you're using Python 3.9 or lower, you'll need to upgrade your Python version to test with Transformers 5.0.

To test with a specific version of transformers (with Python 3.10+):

```bash
pip install transformers==5.0.0
pytest tests/