from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent

setup(
    name="visual_bge",
    version="0.1.0",
    description="visual_bge",
    long_description=(this_dir / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/Solenya-AIaaS/FlagEmbedding/tree/main/research/visual_bge",
    packages=find_packages(include=("visual_bge", "visual_bge.*")),
    package_dir={"": "."},           # base dir is the project root
    include_package_data=True,       # keep model configs, vocab etc.
    install_requires=[
        "torchvision",
        "timm",
        "einops",
        "ftfy",
    ],
    python_requires=">=3.8",
)
