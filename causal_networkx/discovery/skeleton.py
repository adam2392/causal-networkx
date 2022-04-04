from typing import Callable, Set, Dict
from itertools import combinations, permutations

import networkx as nx
import pandas as pd


def learn_skeleton_graph(
    X: pd.DataFrame,
    graph: nx.Graph,
    fixed_edges: nx.Graph,
    sep_set: Dict[Dict[Set]],
    ci_estimator: Callable,
    alpha: float = 0.05,
    max_cond_set_size: int = None,
    **ci_estimator_kwargs,
) -> nx.Graph:
    """Learn a graph from data.

    Parameters
    ----------
    X : pandas.DataFrame
        A dataframe consisting of nodes as columns
        and samples as rows.
    graph : networkx.Graph
        The initialized graph. By default a complete graph.
    fixed_edges : set
        The set of fixed edges.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Returns
    -------
    graph : networkx.Graph
        The discovered graph from data. Stored using an undirected
        graph.
    """
    nodes = graph.nodes
    size_cond_set = 0

    while 1:
        cont = False
        remove_edges = []

        # loop through all possible permutation of
        # two nodes in the graph
        for (i, j) in permutations(nodes, 2):
            # ignore fixed edges
            if (i, j) in fixed_edges:
                continue

            # check that neighbors for "i" contain "j"
            adj_i = list(graph.neighbors(i))
            if j not in adj_i:
                continue
            adj_i.remove(j)

            # check that number of adjacencies is greater then the
            # cardinality of the conditioning set
            if len(adj_i) >= size_cond_set:
                # loop through all possible conditioning sets of certain size
                for cond_set in combinations(adj_i, size_cond_set):
                    # compute conditional independence test
                    _, pvalue = ci_estimator(X, i, j, set(cond_set), **ci_estimator_kwargs)

                    # two variables found to be independent given a separating set
                    if pvalue > alpha:
                        if graph.has_edge(i, j):
                            remove_edges.append((i, j))

                        sep_set[i][j] |= set(cond_set)
                        sep_set[j][i] |= set(cond_set)
                        break
                cont = True
        size_cond_set += 1

        # finally remove edges after performing
        # conditional independence tests
        graph.remove_edges_from(remove_edges)

        # determine if we reached the maximum number of conditioning,
        # or we pruned all possible permutations of nodes
        if size_cond_set > max_cond_set_size or cont is False:
            break

    return graph, sep_set
