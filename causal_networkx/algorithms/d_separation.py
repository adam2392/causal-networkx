from typing import Union

import numpy as np
from networkx.algorithms import d_separated as nx_d_separated

from causal_networkx import ADMG, DAG


def d_separated(G: Union[DAG, ADMG], x, y, z=None):
    """Check d-separation among 'x' and 'y' given 'z' in graph G.

    This algorithm wraps ``networkx.algorithms.d_separated``, but
    allows one to pass in a ``ADMG`` instance instead.

    It first converts all bidirected edges into explicit unobserved
    confounding nodes in an explicit ``networkx.DiGraph``, which then
    calls ``networkx.algorithms.d_separated`` to determine d-separation.
    This inherently increases the runtime cost if there are many
    bidirected edges, because many nodes must be added.

    Parameters
    ----------
    G : ADMG
        Causal graph.
    x : set
        First set of nodes in ``G``.
    y : set
        Second set of nodes in ``G``.
    z : set
        Set of conditioning nodes in ``G``. Can be empty set.

    See Also
    --------
    causal_networkx.ADMG
    networkx.algorithms.d_separation.d_separated

    Notes
    -----
    This wraps the networkx implementation, which only allows DAGs. Since
    ``ADMG`` is not represented.

    """
    if z is None:
        z = set()

    # run d-separation
    if isinstance(x, np.ndarray):
        x = set(list(x))
    elif isinstance(x, str):
        x = set([x])
    elif type(x) == int or float:
        x = set([x])

    if isinstance(y, np.ndarray):
        y = set(list(y))
    elif isinstance(y, str):
        y = set([y])
    elif type(y) == int or float:
        y = set([y])
    if isinstance(z, np.ndarray):
        z = set(list(z))
    elif isinstance(z, str):
        z = set([z])
    elif type(z) in (int, float):
        z = set([z])

    if type(G).__name__ == "DAG":
        return nx_d_separated(G.to_networkx(), x, y, z)

    # get the full graph by converting bidirected edges into latent confounders
    # and keeping the directed edges
    explicit_G = G.compute_full_graph(to_networkx=True)

    # make sure there are always conditioned on the conditioning set
    z = z.union(G._cond_set)
    return nx_d_separated(explicit_G, x, y, z)
