# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../"))

from causal_networkx.version import VERSION, VERSION_SHORT  # noqa: E402

curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "causal_networkx")))
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))

# -- Project information -----------------------------------------------------

project = "Causal-Networkx"
copyright = f"{datetime.today().year}, Adam Li"
author = "Adam Li"
version = VERSION_SHORT
release = VERSION

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "numpydoc",
    'nbsphinx',
    "gh_substitutions",
]

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# generate autosummary even if no references
# -- sphinx.ext.autosummary
autosummary_generate = True

autodoc_default_options = {"inherited-members": None}
autodoc_typehints = "signature"

# -- numpydoc
# Below is needed to prevent errors
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True
numpydoc_validate = True

numpydoc_xref_ignore = {
    # words
    'instance', 'instances', 'of', 'default', 'shape', 'or',
    'with', 'length', 'pair', 'matplotlib', 'optional', 'kwargs', 'in',
    'dtype', 'object', 'self.verbose', 'py', 'the', 'functions', 'lambda',
    'container', 'iterator', 'keyword', 'arguments', 'no', 'attributes',
    # networkx
    'node', 'nodes', 'graph',
    # shapes
    'n_times', 'obj', 'arrays', 'lists', 'func', 'n_nodes',
    'n_estimated_nodes', 'n_samples', 'n_variables',
}
numpydoc_xref_aliases = {
    # Networkx
    'nx.Graph': 'networkx.Graph', 'nx.DiGraph': 'networkx.DiGraph',
    # Causal-Networkx
    'CausalGraph': 'causal_networkx.CausalGraph', 'PAG': 'causal_networkx.PAG',
    # joblib
    'joblib.Parallel': 'joblib.Parallel',
    # pandas
    'pd.DataFrame': 'pandas.DataFrame', 'pandas.DataFrame': 'pandas.DataFrame',
}

default_role = 'py:obj'

# Tell myst-parser to assign header anchors for h1-h3.
# myst_heading_anchors = 3
# suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "networkx": ("https://networkx.org/documentation/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    # Uncomment these if you use them in your codebase:
    #  "torch": ("https://pytorch.org/docs/stable", None),
    #  "datasets": ("https://huggingface.co/docs/datasets/master/en", None),
    #  "transformers": ("https://huggingface.co/docs/transformers/master/en", None),
}
intersphinx_timeout = 5

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Clean up sidebar: Do not show "Source" link
# html_show_sourcelink = False
# html_copy_source = False

html_theme = "pydata_sphinx_theme"

html_title = f"causal-networkx v{VERSION}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
templates_path = ['_templates']
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = "_static/favicon.ico"

html_theme_options = {
    'icon_links': [
        dict(name='GitHub',
             url='https://github.com/adam2392/causal-networkx',
             icon='fab fa-github-square'),
    ],
    'use_edit_page_button': False,
    'navigation_with_keys': False,
    'show_toc_level': 1,
    'navbar_end': ['version-switcher', 'navbar-icon-links'],
}

scrapers = ('matplotlib',)

# sphinx_gallery_conf = {
#     'doc_module': 'causal_networkx',
#     'reference_url': {
#         'causal_networkx': None,
#     },
#     'backreferences_dir': 'generated',
#     'plot_gallery': 'True',  # Avoid annoying Unicode/bool default warning
#     'within_subsection_order': ExampleTitleSortKey,
#     'examples_dirs': ['../examples'],
#     'gallery_dirs': ['auto_examples'],
#     'filename_pattern': '^((?!sgskip).)*$',
#     'matplotlib_animations': True,
#     'compress_images': ('images', 'thumbnails'),
#     'image_scrapers': scrapers,
# }

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'index': ['search-field.html'],
}

html_context = {
    'versions_dropdown': {
        'dev': 'v0.1 (devel)',
    },
}

# Enable nitpicky mode - which ensures that all references in the docs
# resolve.

nitpicky = True
nitpick_ignore = []