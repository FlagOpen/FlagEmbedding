import json
from pathlib import Path


def save_sentence_transformers_config(
    output_dir: str,
    hidden_size: int,
    pooling_method: str,
    normalize_embeddings: bool,
) -> None:
    """Save the dense embedding pipeline alongside the saved transformer weights."""
    output_path = Path(output_dir)
    pooling_path = output_path / "1_Pooling"
    pooling_path.mkdir(exist_ok=True)
    pooling_mode = "lasttoken" if pooling_method == "last_token" else pooling_method
    with (pooling_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"word_embedding_dimension": hidden_size, "pooling_mode": pooling_mode},
            f,
            indent=2,
        )

    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    if normalize_embeddings:
        (output_path / "2_Normalize").mkdir(exist_ok=True)
        modules.append(
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )

    with (output_path / "modules.json").open("w", encoding="utf-8") as f:
        json.dump(modules, f, indent=2)
    with (output_path / "config_sentence_transformers.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({"prompts": {}, "default_prompt_name": None}, f, indent=2)
