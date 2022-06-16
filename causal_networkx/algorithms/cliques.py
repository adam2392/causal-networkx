from typing import Iterator

from causal_networkx import ADMG


def find_cliques(G: ADMG, nodes=None) -> Iterator:
    """Find all maximal cliques in causal DAG.

    This operates over the directed edges of the causal graph, excluding
    the bidirected edges.

    Parameters
    ----------
    G : ADMG
        The causal diagram.
    nodes : list, optional
        The list of nodes to consider, by default None

    Returns
    -------
    iterator
        The cliques in a causal DAG.
    """
    from networkx.algorithms import find_cliques as nx_find_cliques

    return nx_find_cliques(G.dag, nodes=nodes)
