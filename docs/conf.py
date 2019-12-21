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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'plaidml'
copyright = '2019, Intel Corporation'
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
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

extensions = [
    'breathe',
    'sphinx_rtd_theme',
    'sphinxcontrib.katex',
    # 'exhale',
]

# Setup the breathe extension
breathe_projects = {
    "PlaidML": "./xml",
    # "edsl": "./xml",
}

breathe_default_project = "PlaidML"

# breathe_projects_source = {
#     "edsl": ("../plaidml2/edsl", ["edsl.h"]),
# }

# breathe_doxygen_config_options = {
#     'HIDE_FRIEND_COMPOUNDS': 'YES',
#     'HIDE_UNDOC_CLASSES': 'YES',
#     'HIDE_UNDOC_MEMBERS': 'YES',
# }

# Setup the exhale extension
# exhale_args = {
#     # These arguments are required
#     "containmentFolder":
#         "./exhale",
#     "rootFileName":
#         "plaidml2.rst",
#     "rootFileTitle":
#         "PlaidML2 API",
#     "afterTitleDescription":
#         "PlaidML is an open source, extensible deep learning tensor compiler.",
#     "doxygenStripFromPath":
#         "..",
#     # Suggested optional arguments
#     "createTreeView":
#         True,
#     # TIP: if using the sphinx-bootstrap-theme, you need
#     # "treeViewIsBootstrap": True,
#     "exhaleExecutesDoxygen":
#         True,
#     "exhaleUseDoxyfile":
#         True,
#     # "exhaleDoxygenStdin":
#     #     "INPUT = ../plaidml2"
# }

# Tell sphinx what the primary language being documented is.
# primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
# highlight_language = 'cpp'
