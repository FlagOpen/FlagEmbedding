import os
from typing import Collection


def resolve_model_name(model_name_or_path: str, model_names: Collection[str]) -> str:
    """Resolve a model mapping key from a Hub ID or local model path."""
    model_path = os.path.normpath(model_name_or_path)
    if os.path.basename(model_path).startswith("checkpoint-"):
        model_path = os.path.dirname(model_path)

    for candidate in (model_path, os.path.basename(model_path)):
        if candidate in model_names:
            return candidate

    local_name = os.path.basename(model_path)
    matching_names = [
        name for name in model_names if os.path.basename(name) == local_name
    ]
    if len(matching_names) == 1:
        return matching_names[0]

    return local_name
