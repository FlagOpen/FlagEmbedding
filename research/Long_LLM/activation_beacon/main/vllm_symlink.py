import json
import os
import pathlib
import shutil
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_folder", type=str)
    args = parser.parse_args()
    
    folder = args.model_folder.rstrip(os.sep)
    path = pathlib.Path(folder)
    parent = path.parent
    name = path.name
    folder_extrapolation = os.path.join(parent, f"extrapolation-{name}")
    folder_yarn_4 = os.path.join(parent, f"yarn-4-{name}")
    folder_yarn_8 = os.path.join(parent, f"yarn-8-{name}")

    if os.path.exists(folder_extrapolation):
        shutil.rmtree(folder_extrapolation)
    if os.path.exists(folder_yarn_4):
        shutil.rmtree(folder_yarn_4)
    if os.path.exists(folder_yarn_8):
        shutil.rmtree(folder_yarn_8)

    os.makedirs(folder_extrapolation)
    os.makedirs(folder_yarn_4)
    os.makedirs(folder_yarn_8)

    for name in os.listdir(folder):
        if name == "config.json":
            with open(os.path.join(folder, name), "r", encoding="utf-8") as f:
                config = json.load(f)

            extrapolation_config = config.copy()
            extrapolation_config["max_position_embeddings"] = extrapolation_config["max_position_embeddings"] * 8
            if "sliding_window" in extrapolation_config and extrapolation_config["sliding_window"] is not None:
                extrapolation_config["sliding_window"] = extrapolation_config["max_position_embeddings"]
            with open(os.path.join(folder_extrapolation, name), "w", encoding="utf-8") as f:
                json.dump(extrapolation_config, f)

            yarn_4_config = config.copy()
            yarn_4_config["rope_scaling"] = {
                "type": "yarn",
                "factor": 4,
                "original_max_position_embeddings": yarn_4_config["max_position_embeddings"]
            }
            with open(os.path.join(folder_yarn_4, name), "w", encoding="utf-8") as f:
                json.dump(yarn_4_config, f)
            
            yarn_8_config = config.copy()
            yarn_8_config["rope_scaling"] = {
                "type": "yarn",
                "factor": 8,
                "original_max_position_embeddings": yarn_8_config["max_position_embeddings"]
            }
            with open(os.path.join(folder_yarn_8, name), "w", encoding="utf-8") as f:
                json.dump(yarn_8_config, f)

        else:
            src = os.path.join(folder, name)

            dest = os.path.join(folder_extrapolation, name)
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(src, dest)

            dest = os.path.join(folder_yarn_4, name)
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(src, dest)
            
            dest = os.path.join(folder_yarn_8, name)
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(src, dest)
