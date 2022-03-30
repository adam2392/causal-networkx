from typing import Iterable, Iterator
from typing import Iterator
from causal_networkx.cgm import CausalGraph


def find_cliques(G: CausalGraph, nodes=None) -> Iterator:
    """Find all maximal cliques in causal DAG.

    Parameters
    ----------
    G : CausalGraph
        _description_
    nodes : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Yields
    ------
    Iterator
        _description_
    """
    from networkx.algorithms import find_cliques as nx_find_cliques

    return nx_find_cliques(G.dag, nodes=nodes)
