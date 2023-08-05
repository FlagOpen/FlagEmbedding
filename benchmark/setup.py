from setuptools import setup, find_packages

setup(
    name='C_MTEB',
    version='1.0.0',
    author_email='2906698981@qq.com',
    url='https://github.com/FlagOpen/FlagEmbedding/tree/master/benchmark',
    package_dir={"": "C_MTEB"},
    packages=find_packages("C_MTEB"),
    install_requires=[
        'mteb[beir]',
    ],
)
