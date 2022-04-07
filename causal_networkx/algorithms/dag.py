from networkx.algorithms import (
    is_directed_acyclic_graph as nx_is_directed_acyclic_graph,
)
from networkx.algorithms import topological_sort as nx_topological_sort

from causal_networkx.cgm import CausalGraph


def is_directed_acyclic_graph(G):
    """Check if `G` is a directed acyclic graph (DAG) or not.

    Parameters
    ----------
    G : CausalGraph

    Returns
    -------
    bool
        True if `G` is a DAG, False otherwise

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

    Raises
    ------
    NetworkXError
        Topological sort is defined for directed graphs only. If the graph `G`
        is undirected, a :exc:`NetworkXError` is raised.

    NetworkXUnfeasible
        If `G` is not a directed acyclic graph (DAG) no topological sort exists
        and a :exc:`NetworkXUnfeasible` exception is raised.  This can also be
        raised if `G` is changed while the returned iterator is being processed

    RuntimeError
        If `G` is changed while the returned iterator is being processed.

    Notes
    -----
    This algorithm is based on a description and proof in
    "Introduction to Algorithms: A Creative Approach" [1]_ .

    See also
    --------
    networkx.algorithms.topological_sort

    References
    ----------
    .. [1] Manber, U. (1989).
       *Introduction to Algorithms - A Creative Approach.* Addison-Wesley.
    """
    assert isinstance(G, CausalGraph)

    # topological sorting only occurs from the directed edges,
    # not bi-directed edges
    return nx_topological_sort(G.dag)
