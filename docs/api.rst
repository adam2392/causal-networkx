###
API
###

:py:mod:`causal_networkx`:

.. automodule:: causal_networkx
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of Causal-Networkx, grouped thematically by analysis
stage.

Most-used classes
=================
These are the causal classes for Structural Causal Models (SCMs), or various causal
graphs encountered in the literature. 

.. currentmodule:: causal_networkx

.. autosummary::
   :toctree: generated/

   StructuralCausalModel
   DAG
   CPDAG
   ADMG
   PAG

To see a breakdown of different inner graph functionalities,
see the :ref:`Graph API <graph_api>` page. See 

.. toctree::
   :maxdepth: 0

   graph_api

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
   RobustPC

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

   BaseConditionalIndependenceTest
   GSquareCITest
   FisherZCITest
   KernelCITest
   PartialCorrelation
   Oracle
   ParentOracle


IO for reading/writing causal graphs
====================================
We advocate for using our implemented causal graph classes whenever
utilizing various packages. However, we also support transformations
to and from popular storage classes, such as ``numpy arrays``,
``pandas dataframes``, ``pgmpy``, ``DOT`` and ``dagitty``. Note that
not all these are supported for all types of graphs because of
inherent limitations in supporting mixed-edge graphs in other formats.

.. currentmodule:: causal_networkx.io

.. autosummary::
   :toctree: generated/

   load_from_networkx
   load_from_numpy
   load_from_pgmpy
   to_networkx
   to_numpy


Utility Algorithms for Causal Graphs
====================================
.. currentmodule:: causal_networkx.algorithms

.. autosummary::
   :toctree: generated/

   d_separated
   find_cliques
   is_directed_acyclic_graph
   topological_sort
   discriminating_path
   possibly_d_sep_sets
   uncovered_pd_path
   dag2cpdag
   admg2pag
   is_markov_equivalent
   compute_v_structures