from itertools import combinations
from typing import Set, Tuple

from networkx.algorithms import (
    is_directed_acyclic_graph as nx_is_directed_acyclic_graph,
)
from networkx.algorithms import topological_sort as nx_topological_sort

from causal_networkx import ADMG, DAG


def is_directed_acyclic_graph(G) -> bool:
    """Check if ``G`` is a directed acyclic graph (DAG) or not.

    Parameters
    ----------
    G : ADMG

    Returns
    -------
    is_dag : bool
        True if ``G`` is a DAG, False otherwise

    Examples
    --------
    Undirected graph::

        >>> G = ADMG(nx.Graph([(1, 2), (2, 3)]))
        >>> causal_networkx.is_directed_acyclic_graph(G)
        False

    Directed graph with cycle::

        >>> G = ADMG(nx.DiGraph([(1, 2), (2, 3), (3, 1)]))
        >>> causal_networkx.is_directed_acyclic_graph(G)
        False

    Directed acyclic graph::

        >>> G = ADMG(nx.DiGraph([(1, 2), (2, 3)]))
        >>> causal_networkx.is_directed_acyclic_graph(G)
        True

    See also
    --------
    topological_sort
    """
    return nx_is_directed_acyclic_graph(G.dag)


def topological_sort(G: ADMG):
    """Returns a generator of nodes in topologically sorted order.

    A topological sort is a nonunique permutation of the nodes of a
    directed graph such that an edge from u to v implies that u
    appears before v in the topological sort order. This ordering is
    valid only if the graph has no directed cycles.

    Parameters
    ----------
    G : ADMG
        A causal directed acyclic graph (DAG)

    Yields
    ------
    nodes
        Yields the nodes in topological sorted order.

    See also
    --------
    networkx.algorithms.dag.topological_sort
    """
    assert isinstance(G, ADMG)

    # topological sorting only occurs from the directed edges,
    # not bi-directed edges
    return nx_topological_sort(G.dag)


def compute_v_structures(graph: DAG) -> Set[Tuple]:
    """Iterate through the graph to compute all v-structures.

    Parameters
    ----------
    graph : instance of DAG
        A causal graph.

    Returns
    -------
    vstructs : Set[Tuple]
        The v structures within the graph.
    """
    vstructs: Set[Tuple] = set()
    for node in graph.nodes:
        for p1, p2 in combinations(graph.parents(node) | graph.spouses(node), 2):
            if not graph.has_adjacency(p1, p2):
                p1_, p2_ = sorted((p1, p2))
                vstructs.add((p1_, node, p2_))
    return vstructs
