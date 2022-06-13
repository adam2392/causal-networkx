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

from causal_networkx import StructuralCausalModel

# %%
# Structural Causal Models: Simulating some data
# ----------------------------------------------
#
# Structural causal models (SCMs) are mathematical objects defined
# by a 4-tuple <V, U, F, P(U)>, where:
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
func_uz = lambda: rng.binomial(n=1, p=0.25)
func_ux = lambda: rng.binomial(n=1, p=0.25)
func_uxy = lambda: rng.binomial(n=1, p=0.4)
func_xy = lambda u_xy: u_xy
func_x = lambda u_x: 2 * u_x
func_y = lambda x, u_xy, z: x * u_xy + z
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
print(G.parents("y"))

# Or the children of 'xy'
print(G.children("xy"))

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
print(admg.parents("y"))

# The bidirected edges also form a cluster in what is known as "confounded-components", or
# c-components for short.
print(admg.c_components)

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
#
#

# %%
# Partial Ancestral Graph (PAG)
# -----------------------------
#
#
