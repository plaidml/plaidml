# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'plaidml'
copyright = '2020, Intel Corporation'
author = 'plaidml'

# -- General configuration ---------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

extensions = [
    'breathe',
    'sphinx_rtd_theme',
    'sphinx_tabs.tabs',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
]

autodoc_mock_imports = [
    'plaidml.core._version',
    'plaidml.ffi',
]

autosectionlabel_prefix_document = True

# Setup the breathe extension
breathe_projects = {
    'PlaidML': './xml',
}

breathe_default_project = 'PlaidML'

pygments_style = 'sphinx'
# breathe_doxygen_config_options = {
#     'HIDE_FRIEND_COMPOUNDS': 'YES',
#     'HIDE_UNDOC_CLASSES': 'YES',
#     'HIDE_UNDOC_MEMBERS': 'YES',
# }

# Tell sphinx what the primary language being documented is.
# primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
# highlight_language = 'cpp'

if os.environ.get('READTHEDOCS') == 'True':
    import subprocess
    import util
    subprocess.run(['doxygen'])
    util.fix_doxyxml('xml/*.xml')
