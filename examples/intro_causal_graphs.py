"""
.. _ex-causal-graphs:

====================================================
An introduction to causal graphs and how to use them
====================================================

Causal graphs are graphical objects that attach causal notions to each edge
and missing edge. We will review some of the fundamental causal graphs used
in causal inference, and their differences using implementations in ``causal-networkx``.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from causal_networkx import CPDAG, PAG, StructuralCausalModel
from causal_networkx.algorithms import d_separated
from causal_networkx.ci import Oracle
from causal_networkx.discovery import PC

# %%
# Structural Causal Models: Simulating some data
# ----------------------------------------------
#
# Structural causal models (SCMs) :footcite:`Pearl_causality_2009` are mathematical objects
# defined by a 4-tuple <V, U, F, P(U)>, where:
#
#   - V is the set of endogenous observed variables
#   - U is the set of exogenous unobserved variables
#   - F is the set of functions for every $v \in V$
#   - P(U) is the set of distributions for all $u \in U$
#
# Taken together, these four objects define the generating causal
# mechanisms for a causal problem. Almost always, the SCM is not known.
# However, the SCM induces a causal graph, which has nodes from ``V``
# and then edges are defined by the arguments of the functions in ``F``.
# If there are common exogenous parents for any V, then this can be represented
# in an Acyclic Directed Mixed Graph (ADMG), or a causal graph with bidirected edges.
# The common latent confounder is represented with a bidirected edge between two
# endogenous variables.
#
# Even though the SCM is typically unknown, we can still use it to generate
# ground-truth simulations to evaluate various algorithms and build our intuition.
# Here, we will simulate some data to understand causal graphs in the context of SCMs.

# set a random seed to make example reproducible
seed = 12345
rng = np.random.RandomState(seed=seed)

# construct a SCM with 2 exogenous variables and 4 endogenous variables
# x, xy, z -> y; xy -> x
func_uz = lambda: rng.binomial(n=1, p=0.25)
func_ux = lambda: rng.binomial(n=1, p=0.25)
func_uxy = lambda: rng.binomial(n=1, p=0.4)
func_xy = lambda u_xy: u_xy
func_x = lambda u_x, xy: 2 * u_x + xy
func_y = lambda x, xy, z: x * xy + z
func_z = lambda u_z: u_z

# construct the SCM
scm = StructuralCausalModel(
    exogenous={
        "u_xy": func_uxy,
        "u_z": func_uz,
        "u_x": func_ux,
    },
    endogenous={"x": func_x, "y": func_y, "z": func_z, "xy": func_xy},
)

# sample the incomplete observational data
data = scm.sample(n=5000, include_latents=False)

# We can now get the induced causal graph, which is a causal DAG
# in this case, since there are no exogenous confounders.
# We then say the SCM is "Markovian".
G = scm.get_causal_graph()

# note the graph shows colliders, which is a collision of arrows
# for example between ``x`` and ``z`` at ``y``.
G.draw()

# %%
# Causal Directed Ayclic Graphs (DAG): Also known as Causal Bayesian Networks
# ---------------------------------------------------------------------------
#
# Causal DAGs represent Markovian SCMs, also known as the "causally sufficient"
# assumption, where there are no unobserved confounders in the graph.
print(G)

# One can query the parents of 'y' for example
print(list(G.parents("y")))

# Or the children of 'xy'
print(list(G.children("xy")))

# Using the graph, we can explore d-separation statements, which by the Markov
# condition, imply conditional independences.
# For example, 'z' is d-separated from 'x' because of the collider at 'y'
print(f"'z' is d-separated from 'x': {d_separated(G, 'z', 'x')}")

# Conditioning on the collider, opens up the path
print(f"'z' is d-separated from 'x' given 'y': {d_separated(G, 'z', 'x', 'y')}")

# %%
# Acyclic Directed Mixed Graphs (ADMG)
# ------------------------------------
#
# ADMGs represent Semi-Markovian SCMs, where there are possibly unobserved confounders
# in the graph. These unobserved confounders are graphically depicted with a bidirected edge.

# We can construct an ADMG from the DAG by just setting 'xy' as a latent confounder
admg = G.set_nodes_as_latent_confounders(["xy"])

# Now there is a bidirected edge between 'x' and 'y'
admg.draw()

# Now if one queries the parents of 'y', it will not show 'xy' anymore
print(list(admg.parents("y")))

# The bidirected edges also form a cluster in what is known as "confounded-components", or
# c-components for short.
print(f"The ADMG has c-components: {admg.c_components}")

# We can also look at d-separation statements similarly to a DAG.
# For example, 'z' is still d-separated from 'x' because of the collider at 'y'
print(f"'z' is d-separated from 'x': {d_separated(admg, 'z', 'x')}")

# Conditioning on the collider, opens up the path
print(f"'z' is d-separated from 'x' given 'y': {d_separated(admg, 'z', 'x', 'y')}")

# Say we add a bidirected edge between 'z' and 'x', then they are no longer
# d-separated.
admg.add_bidirected_edge("z", "x")
print(f"'z' is d-separated from 'x': {d_separated(admg, 'z', 'x')}")

# Markov Equivalence Classes
# --------------------------
#
# Besides graphs that represent causal relationships from the SCM, there are other
# graphical objects used in the causal inference literature.
#
# Markov equivalence class graphs are graphs that encode the same Markov equivalences
# or d-separation statements, or conditional independences. These graphs are commonly
# used in constraint-based structure learning algorithms, which seek to reconstruct
# parts of the causal graph from data. In this next section, we will briefly overview
# some of these common graphs.

# %%
# Completed Partially Directed Ayclic Graph (CPDAG)
# -------------------------------------------------
# CPDAGs are Markov Equivalence class graphs that encode the same d-separation statements
# as a causal DAG that stems from a Markovian SCM. All relevant variables are assumed to
# be observed. An uncertain edge orientation is encoded via a undirected edge between two
# variables. Here, we'll construct a CPDAG that encodes the same d-separations as the
# earlier DAG.
#
# Typically, CPDAGs are learnt using some variant of the PC algorithm :footcite:`Spirtes1993`.

cpdag = CPDAG()

# let's assume all the undirected edges are formed from the earlier DAG
cpdag.add_undirected_edges_from(G.edges)

# next, we will orient all unshielded colliders present in the original DAG
cpdag.orient_undirected_edge("x", "y")
cpdag.orient_undirected_edge("xy", "y")
cpdag.orient_undirected_edge("z", "y")

# Note: the CPDAG orients all edges that participate in an unshielded collider.
# The only non-oriented edge is between 'xy' and 'x'. We can
# run the full PC algorithm using the oracle graph, and verify the output.
pc = PC(ci_estimator=Oracle(G))
pc.fit(data)
graph_ = pc.graph_

# The two graphs match exactly
print(graph_.all_edges())
print(cpdag.all_edges())

# %%
# Partial Ancestral Graph (PAG)
# -----------------------------
# PAGs are Markov equivalence classes for ADMGs. Since we allow latent confounders, these graphs
# are more complex compared to the CPDAGs. PAGs encode uncertain edge orientations via circle
# endpoints. A circle endpoint (``o-*``) can imply either: a tail (``-*``), or an arrowhead (``<-*``),
# which can then imply either an undirected edge (selection bias), directed edge (ancestral relationship),
# or bidirected edge (possible presence of a latent confounder).
#
# Note: a directed edge in the PAG does not actually imply parental relationships.
#
# Typically, PAGs are learnt using some variant of the FCI algorithm :footcite:`Spirtes1993` and
# :footcite`Zhang2008`.
pag = PAG()

# %%
# References
# ^^^^^^^^^^
# .. footbibliography::
