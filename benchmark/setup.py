from setuptools import setup, find_packages

setup(
    name='C_MTEB',
    version='1.0.0',
    package_dir={"": "C_MTEB"},
    packages=find_packages("C_MTEB"),
    install_requires=[
        'mteb[beir]',
    ],
)
