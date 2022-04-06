from typing import Iterable
import numpy as np
import networkx as nx
from networkx.algorithms import d_separated as nx_d_separated

from causal_networkx.cgm import CausalGraph
from causal_networkx.utils import convert_latent_to_unobserved_confounders


def d_separated(G: CausalGraph, x, y, z):
    """Check d-separation among 'x' and 'y' given 'z' in graph G.

    This algorithm wraps `networkx.algorithms.d_separated`, but
    allows one to pass in a `CausalGraph` instance instead.

    It first converts all bidirected edges into explicit unobserved
    confounding nodes in an explicit `networkx.DiGraph`, which then
    calls `networkx.algorithms.d_separated` to determine d-separation.
    This inherently increases the runtime cost if there are many
    bidirected edges, because many nodes must be added.

    Parameters
    ----------
    G : CausalGraph
        Causal graph.
    x : set
        First set of nodes in ``G``.
    y : set
        Second set of nodes in ``G``.
    z : set
        Set of conditioning nodes in ``G``. Can be empty set.

    See Also
    --------
    causal_networkx.CausalGraph
    networkx.algorithms.d_separated

    Notes
    -----
    This wraps the networkx implementation, which only allows DAGs. Since
    ``CausalGraph`` is not represented.
    """
    # get the full graph
    explicit_G = G.compute_full_graph()

    # run d-separation
    if isinstance(x, np.ndarray):
        x = set(list(x))
    elif isinstance(x, str):
        x = set([x])
    if isinstance(y, np.ndarray):
        y = set(list(y))
    elif isinstance(y, str):
        y = set([y])
    if isinstance(z, np.ndarray):
        z = set(list(z))
    elif isinstance(z, str):
        z = set([z])

    return nx_d_separated(explicit_G, x, y, z)
