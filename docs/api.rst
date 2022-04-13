###
API
###

:py:mod:`causal_networkx`:

.. automodule:: causal_networkx
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-Connectivity, grouped thematically by analysis
stage. The data structure classes contain different types of connectivity data
and are described below.

Most-used classes
=================

.. currentmodule:: causal_networkx

.. autosummary::
   :toctree: generated/

   StructuralCausalModel
   CausalGraph
   PAG

Discovery functions
===================

These functions compute try to learn the causal graph from data. 
All these functions work with a Pandas ``DataFrame`` object,
which is the recommended input to these functions. However, they
also should work on numpy array inputs.

.. currentmodule:: causal_networkx.discovery
    
.. autosummary::
   :toctree: generated/

   PC
   FCI

Conditional independence testing functions
==========================================

These functions are implementations of common conditional independence (CI) testing
functions. These will eventually be ported out of this repository as they are of
independent interest.

The general API for CI tests require a ``data``, ``x``, ``y`` and ``sep_set``
input, where ``data`` is the DataFrame containing the data to be analyzed,
``x`` and ``y`` are single columns of the DataFrame corresponding to a single
variable and then ``sep_set`` is the conditioning set of variables.

.. currentmodule:: causal_networkx.ci

.. autosummary::
   :toctree: generated/

   g_square_binary
   g_square_discrete
   fisherz
   Oracle

Utility Algorithms for Causal Graphs
====================================
.. currentmodule:: causal_networkx.algorithms

.. autosummary::
   :toctree: generated/

   d_separated
   find_cliques
   is_directed_acyclic_graph
   topological_sort