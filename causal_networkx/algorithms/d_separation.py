from typing import Set, Union

import networkx as nx
import numpy as np
from networkx.algorithms import d_separated as nx_d_separated

from causal_networkx import ADMG, DAG
from causal_networkx.algorithms.dag import moralize_graph
from causal_networkx.graphs.base import BaseGraph


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


def compute_minimal_separating_set(G: BaseGraph, x, y):
    """Compute the minimal separating set between X and Y.

    Uses the algorithm presented in :footcite:`van-der-zander20a`, `Tian1998FindingMD`.

    Parameters
    ----------
    G : BaseGraph
        The causal graph, which provides an oracle for d-separation.
    x : node
        Node X in the graph, G.
    y : node
        Node Y in the graph, G.

    References
    ----------
    .. footbibliography::
    """
    # first construct the ancestors of X and Y
    x_anc = G.ancestors(x)
    y_anc = G.ancestors(y)
    D_anc_xy = x_anc.union(y_anc)
    D_anc_xy = D_anc_xy.union((x, y))

    # second, construct the moralization of the subgraph
    moral_G = moralize_graph(G.subgraph(D_anc_xy))

    # find a separating set Z' in moral_G
    # TODO: make work for ADMG too
    Z_prime = set(G.parents(x)).union(set(G.parents(y)))
    assert d_separated(G, x, y, Z_prime)

    # perform BFS on the graph from 'x' to mark
    Z_dprime = _bfs_with_marks(moral_G, x, Z_prime)
    Z = _bfs_with_marks(moral_G, y, Z_dprime)
    return Z


def is_separating_set_minimal(G: nx.DiGraph, x, y, z: Set) -> bool:
    """Determine if a separating set is minimal.

    Uses the algorithm 2 presented in :footcite:`Tian1998FindingMD`.

    Parameters
    ----------
    G : nx.DiGraph
        The graph.
    x : node
        X node.
    y : node
        Y node.
    z : Set
        The separating set to check is minimal.

    Returns
    -------
    bool
        Whether or not the `z` separating set is minimal.

    References
    ----------
    .. footbibliography::
    """
    x_anc = G.ancestors(x)
    y_anc = G.ancestors(y)
    xy_anc = x_anc.union(y_anc)

    # if Z contains any node which is not in ancestors of X or Y
    # then it is definitely not minimal
    if any(node not in xy_anc for node in z):
        return False

    D_anc_xy = x_anc.union(y_anc)
    D_anc_xy = D_anc_xy.union((x, y))

    # second, construct the moralization of the subgraph
    moral_G = moralize_graph(G.subgraph(D_anc_xy))

    # start BFS from X
    marks = _bfs_with_marks(moral_G, x, z)

    # if not all the Z is marked, then the set is not minimal
    if any(node not in marks for node in z):
        return False

    # similarly, start BFS from Y and check the marks
    marks = _bfs_with_marks(moral_G, y, z)
    # if not all the Z is marked, then the set is not minimal
    if any(node not in marks for node in z):
        return False

    return True


def _bfs_with_marks(G: nx.Graph, start_node, check_set):
    """Breadth-first-search with markings.

    Parameters
    ----------
    G : nx.Graph
        _description_
    start_node : _type_
        _description_
    check_set : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    visited = dict()
    marked = dict()
    queue = []

    visited[start_node] = None
    queue.append(start_node)
    while queue:
        m = queue.pop(0)

        for neighbr in G.neighbors(m):
            if neighbr not in visited:
                # memoize where we visited so far
                visited[neighbr] = None

                # mark the node in Z' and do not continue
                # along that path
                if neighbr in check_set:
                    marked[neighbr] = None
                else:
                    queue.append(neighbr)
    return set(marked.keys())
