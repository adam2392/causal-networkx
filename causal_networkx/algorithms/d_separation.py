from typing import Union

import numpy as np
from networkx.algorithms import d_separated as nx_d_separated

from causal_networkx.cgm import PAG, CausalGraph


def d_separated(G: Union[CausalGraph, PAG], x, y, z):
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
    causal_networkx.algorithms.m_separated
    networkx.algorithms.d_separated

    Notes
    -----
    This wraps the networkx implementation, which only allows DAGs. Since
    ``CausalGraph`` is not represented.

    """
    # get the full graph by converting bidirected edges into latent confounders
    # and keeping the directed edges
    explicit_G = G.compute_full_graph(to_networkx=True)

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

    # make sure there are always conditioned on the conditioning set
    z = z.union(G._cond_set)
    return nx_d_separated(explicit_G, x, y, z)


# from collections import deque

# import networkx as nx
# from networkx.utils import UnionFind

# def m_separated(G, x, y, z):
#     union_xyz = x.union(y).union(z)

#     if any(n not in G.nodes for n in union_xyz):
#         raise nx.NodeNotFound("one or more specified nodes not found in the graph")

#     G_copy = G.copy()

#     # transform the graph by removing leaves that are not in x | y | z
#     # until no more leaves can be removed.
#     leaves = deque([n for n in G_copy.nodes if G_copy.out_degree[n] == 0])
#     while len(leaves) > 0:
#         leaf = leaves.popleft()
#         if leaf not in union_xyz:
#             for p in G_copy.predecessors(leaf):
#                 if G_copy.out_degree[p] == 1:
#                     leaves.append(p)
#             G_copy.remove_node(leaf)

#     # transform the graph by removing outgoing edges from the
#     # conditioning set. This will only remove "directed" edges,
#     # while preserving bidirected and circular edges.
#     edges_to_remove = list(G_copy.out_edges(z))
#     G_copy.remove_edges_from(edges_to_remove)

#     # use disjoint-set data structure to check if any node in `x`
#     # occurs in the same weakly connected component as a node in `y`.
#     disjoint_set = UnionFind(G_copy.nodes())
#     for component in nx.weakly_connected_components(G_copy):
#         disjoint_set.union(*component)
#     disjoint_set.union(*x)
#     disjoint_set.union(*y)

#     if x and y and disjoint_set[next(iter(x))] == disjoint_set[next(iter(y))]:
#         return False
#     else:
#         return True
