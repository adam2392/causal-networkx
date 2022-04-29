from networkx.algorithms import (
    is_directed_acyclic_graph as nx_is_directed_acyclic_graph,
)
from networkx.algorithms import topological_sort as nx_topological_sort

from causal_networkx.cgm import CausalGraph


def is_directed_acyclic_graph(G):
    """Check if ``G`` is a directed acyclic graph (DAG) or not.

    Parameters
    ----------
    G : CausalGraph

    Returns
    -------
    bool
        True if ``G`` is a DAG, False otherwise

    Examples
    --------
    Undirected graph::

        >>> G = CausalGraph(nx.Graph([(1, 2), (2, 3)]))
        >>> causal_networkx.is_directed_acyclic_graph(G)
        False

    Directed graph with cycle::

        >>> G = CausalGraph(nx.DiGraph([(1, 2), (2, 3), (3, 1)]))
        >>> causal_networkx.is_directed_acyclic_graph(G)
        False

    Directed acyclic graph::

        >>> G = CausalGraph(nx.DiGraph([(1, 2), (2, 3)]))
        >>> causal_networkx.is_directed_acyclic_graph(G)
        True

    See also
    --------
    topological_sort
    """
    return nx_is_directed_acyclic_graph(G.dag)


def topological_sort(G: CausalGraph):
    """Returns a generator of nodes in topologically sorted order.

    A topological sort is a nonunique permutation of the nodes of a
    directed graph such that an edge from u to v implies that u
    appears before v in the topological sort order. This ordering is
    valid only if the graph has no directed cycles.

    Parameters
    ----------
    G : CausalGraph
        A causal directed acyclic graph (DAG)

    Yields
    ------
    nodes
        Yields the nodes in topological sorted order.

    See also
    --------
    networkx.algorithms.dag.topological_sort
    """
    assert isinstance(G, CausalGraph)

    # topological sorting only occurs from the directed edges,
    # not bi-directed edges
    return nx_topological_sort(G.dag)
