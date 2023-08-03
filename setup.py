from setuptools import setup, find_packages

setup(
    name='FlagEmbedding',
    version='0.0.1',
    package_dir={"": "flag_embedding"},
    packages=find_packages("flag_embedding"),
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.18.0',
        'datasets',
        'accelerate>=0.20.1'
    ],
)
