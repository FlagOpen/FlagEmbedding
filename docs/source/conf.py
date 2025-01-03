# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

project = 'BGE'
copyright = '2024, BAAI'
author = 'BAAI'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx_design",
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
html_theme = "pydata_sphinx_theme"
html_logo = "_static/img/bge_logo.jpeg"
html_static_path = ['_static']
html_css_files = ["css/custom.css"]

# MyST-NB conf
nb_execution_mode = "off"

html_theme_options = {
    "logo": {
        "text": "BGE",
    },
    "external_links": [
        {
            "url": "https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d",
            "name": "HF Models",
        },
    ],
    "icon_links":[
        {
            "name": "GitHub",
            "url": "https://github.com/FlagOpen/FlagEmbedding",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/FlagEmbedding/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "HF Models",
            "url": "https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d",
            "icon": "fa-solid fa-cube",
        }
    ],
    "navigation_depth": 5,
    "header_links_before_dropdown": 5,
}

html_context = {
   "default_mode": "light"
}