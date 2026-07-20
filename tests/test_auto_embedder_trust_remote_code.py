
"""Registry basename must not silently enable trust_remote_code (#1579)."""

from FlagEmbedding.inference.embedder.model_mapping import AUTO_EMBEDDER_MAPPING
from FlagEmbedding.inference.auto_embedder import FlagAutoModel


def test_registry_has_trust_true_entries():
    # Precondition: at least one mapping still declares trust_remote_code=True
    assert any(getattr(c, "trust_remote_code", False) for c in AUTO_EMBEDDER_MAPPING.values())


def test_from_finetuned_does_not_inherit_registry_trust(monkeypatch):
    """Basename match must not force trust_remote_code=True when omitted."""
    # Pick any mapping entry that wants trust_remote_code=True if present
    name = None
    for k, cfg in AUTO_EMBEDDER_MAPPING.items():
        if getattr(cfg, "trust_remote_code", False):
            name = k
            break
    if name is None:
        # Fallback: any entry
        name = next(iter(AUTO_EMBEDDER_MAPPING))

    captured = {}

    class Dummy:
        DEFAULT_POOLING_METHOD = "cls"

        def __init__(self, model_name_or_path, **kwargs):
            captured.update(kwargs)
            captured["model_name_or_path"] = model_name_or_path

    # Patch the selected model class constructor
    cfg = AUTO_EMBEDDER_MAPPING[name]
    monkeypatch.setattr(cfg, "model_class", Dummy)

    FlagAutoModel.from_finetuned(f"/tmp/evil/{name}")
    assert captured.get("trust_remote_code") is False
