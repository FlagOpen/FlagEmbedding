from packaging import version
import transformers

TF_VER = version.parse(getattr(transformers, "__version__", "0.0.0"))
IS_TF_V5_OR_HIGHER = TF_VER >= version.parse("5.0.0")


# ------------- torch.fx availability -------------
# v5 removed is_torch_fx_available. We emulate it via feature detection.
def is_torch_fx_available():
    try:
        import torch.fx  # noqa: F401

        return True
    except Exception:
        return False


# ------------- other utilities that moved -------------
# Pattern:
# try the new location first (v5), then fall back to v4 path, else provide a safe default.
def import_from_candidates(candidates, default=None):
    for mod, name in candidates:
        try:
            module = __import__(mod, fromlist=[name])
            return getattr(module, name)
        except Exception:
            pass
    return default
