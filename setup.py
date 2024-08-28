from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='FlagEmbedding',
    version='1.2.11',
    description='FlagEmbedding',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='2906698981@qq.com',
    url='https://github.com/FlagOpen/FlagEmbedding',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'FlagEmbedding': [
            'visual/eva_clip/bpe_simple_vocab_16e6.txt.gz',
            'visual/eva_clip/model_configs/*.json'
        ],
    },
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.33.0',
        'datasets',
        'accelerate>=0.20.1',
        'sentence_transformers',
        'peft',
    ],
)
