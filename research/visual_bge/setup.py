from setuptools import setup, find_packages

setup(
    name="visual_bge",
    version="0.1.0",
    description='visual_bge',
    long_description="./README.md",
    long_description_content_type="text/markdown",
    url='https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge',
    packages=find_packages(),
    install_requires=[
        'torchvision',
        'timm',
        'einops',
        'ftfy'
    ],
    python_requires='>=3.6',
)
