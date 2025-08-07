from setuptools import setup, find_packages


setup(
    name="visual_bge",
    version="0.1.0",
    description="visual_bge",
    long_description="README.md",
    long_description_content_type="text/markdown",
    url="https://github.com/Solenya-AIaaS/FlagEmbedding/tree/main/research/visual_bge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,       # keep model configs, vocab etc.
    package_data={
        # every *.gz and *.txt plus YAML configs in this sub-package
        "visual_bge.eva_clip": ["*.gz", "*.txt", "model_configs/*.json"],
    },
    install_requires=[
        "torchvision",
        "timm",
        "einops",
        "ftfy",
    ],
    python_requires=">=3.8",
)
