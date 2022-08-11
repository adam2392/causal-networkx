from itertools import combinations
from typing import Set, Tuple

import networkx as nx
from networkx.algorithms import (
    is_directed_acyclic_graph as nx_is_directed_acyclic_graph,
)
from networkx.algorithms import topological_sort as nx_topological_sort

from causal_networkx import ADMG, DAG


def moralize_graph(G: DAG) -> nx.Graph:
    """Moralize a graph.

    Uses the definition and algorithm presented in :footcite:`Tian1998FindingMD`.

    Parameters
    ----------
    G : instance of DAG
        Causal graph.

    Returns
    -------
    moral_G : nx.Graph
        An undirected graph of the moralization of ``G``.
    """
    # find all v-structures
    v_structs = compute_v_structures(G)
    # add an edge between all parents of common children
    for p1, _, p2 in v_structs:
        G.add_edge(p1, p2)

    # convert graph to undirected graph
    moral_G = G.to_adjacency_graph()
    return moral_G


def is_directed_acyclic_graph(G):
    """Check if ``G`` is a directed acyclic graph (DAG) or not.

    Parameters
    ----------
    G : instance of DAG
        The causal graph.

    Returns
    -------
    is_dag : bool
        True if ``G`` is a DAG, False otherwise.

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
    # topological sorting only occurs from the directed edges,
    # not bi-directed edges
    return nx_topological_sort(G.dag)


def compute_v_structures(graph: DAG) -> Set[Tuple]:
    """Iterate through the graph to compute all v-structures.

    Parameters
    ----------
    graph : instance of DAG | ADMG
        A causal graph.

    Returns
    -------
    vstructs : Set[Tuple]
        The v structures within the graph. Each set has a 3-tuple with the
        parent, collider, and other parent.
    """
    vstructs: Set[Tuple] = set()
    for node in graph.nodes:
        # get a list of the parents and spouses
        parents = set(graph.parents(node))
        spouses = graph.spouses(node)
        triple_candidates = parents.union(spouses)
        for p1, p2 in combinations(triple_candidates, 2):
            if (
                not graph.has_adjacency(p1, p2)  # should be unshielded triple
                and graph.has_edge(p1, node)  # must be connected to the node
                and graph.has_edge(p2, node)  # must be connected to the node
            ):
                p1_, p2_ = sorted((p1, p2))
                vstructs.add((p1_, node, p2_))
    return vstructs
