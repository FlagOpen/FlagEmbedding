from setuptools import setup, find_packages

setup(
    name='flag_embedding',
    version='1.0.0',
    author_email='2906698981@qq.com',
    url='https://github.com/FlagOpen/FlagEmbedding',
    package_dir={"": "flag_embedding"},
    packages=find_packages("flag_embedding"),
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.18.0',
        'datasets',
        'accelerate>=0.20.1'
    ],
)
