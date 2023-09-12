from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='C_MTEB',
    version='1.1.0',
    description='Chinese Massive Text Embedding Benchmark',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='2906698981@qq.com',
    url='https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB',
    packages=find_packages(),
    install_requires=[
        'mteb[beir]',
    ],
)
