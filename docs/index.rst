.. causal_networkx documentation master file, created by
   sphinx-quickstart on Tue Sep 21 08:07:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**causal-networkx**
===================

Causal-Networkx is a Python package for representing causal graphs, such as Acyclic
Directed Mixed Graphs (ADMG), also known as causal DAGs and Partial Ancestral Graphs (PAGs).
We loosely build on top of ``networkx`` such that we maintain all the well-tested and efficient
algorithms and data structures of ``networkx``, and implement causal-specific graph algorithms. 
We implement basic causal discovery algorithms, causal ID algorithms (coming soon) and causal
estimation algorithms (coming soon). It comes with causal graph traversal algorithms,
such as m-separation.

We encourage you to use the package for your causal inference research and also build on top
with relevant Pull Requests.

See our examples for walk-throughs of how to use the package.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   overview
   installation
   api
   whats_new

.. toctree::
   :hidden:
   :caption: Development

   License <https://raw.githubusercontent.com/adam2392/causal-networkx/main/LICENSE>
   Contributing <https://github.com/adam2392/causal-networkx/main/CONTRIBUTING.md>

Team
----

**causal-networkx** is developed and maintained by adam2392.
To learn more about who specifically contributed to this codebase, see
`our contributors <https://github.com/adam2392/causal-networkx/graphs/contributors>`_ page.

License
-------

**causal-networkx** is licensed under `BSD 3.0 <https://opensource.org/licenses/BSD-3-Clause>`_.
A full copy of the license can be found `on GitHub <https://github.com/adam2392/causal-networkx/blob/main/LICENSE>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
