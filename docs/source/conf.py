# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

project = 'FlagEmbedding'
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
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
# html_logo = "_static/img/BAAI_logo.png"
html_title = "FlagEmbedding"
html_static_path = ['_static']
html_theme_options = {
    # "light_logo": "/_static/img/BAAI_logo.png",
    "light_css_variables": {
        "color-brand-primary": "#238be8",
        "color-brand-content": "#238be8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FBCB67",
        "color-brand-content": "#FBCB67",
    },
}

# MyST-NB conf
nb_execution_mode = "off"