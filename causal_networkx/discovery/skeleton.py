from itertools import combinations, permutations
from typing import Callable, Dict, Set

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx.algorithms.pag import possibly_d_sep_sets
from causal_networkx.cgm import PAG


def learn_skeleton_graph_with_pdsep(
    X: pd.DataFrame,
    adj_graph: nx.Graph,
    sep_set: Dict[str, Dict[str, Set]],
    ci_estimator: Callable,
    fixed_edges: Set = None,
    alpha: float = 0.05,
    min_cond_set_size: int = 0,
    max_cond_set_size: int = None,
    max_path_length: int = np.inf,
    pag: PAG = None,
    **ci_estimator_kwargs,
) -> nx.Graph:
    """Learn a graph from data.

    Proceed by testing the possibly d-separating set of nodes.

    Parameters
    ----------
    X : pandas.DataFrame
        A dataframe consisting of nodes as columns
        and samples as rows.
    adj_graph : networkx.Graph
        The initialized graph. Can be for example a complete graph.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    fixed_edges : set
        The set of fixed edges.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    only_neighbors : bool
        Whether to only consider adjacent nodes to the start/end node in the
        conditioning set, or to consider the superset of combinations by computing
        the set of possibly d-separating nodes. Default is False.
    max_path_length : int
        The maximum length of a path to consider when looking for possibly d-separating
        sets among two nodes. Only used if ``only_neighbors=False``. Default is infinite.
    pag : PAG
        The partial ancestral graph. Only used if ``only_neighbors=False``.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Returns
    -------
    graph : networkx.Graph
        The discovered graph from data. Stored using an undirected
        graph.

    See Also
    --------
    causal_networkx.algorithms.possibly_d_sep_sets
    """
    if max_cond_set_size is None:
        max_cond_set_size = np.inf

    nodes = adj_graph.nodes
    size_cond_set = min_cond_set_size

    while 1:
        cont = False
        remove_edges = []

        # loop through all possible permutation of
        # two nodes in the graph
        for (i, j) in permutations(nodes, 2):
            # ignore fixed edges
            if (i, j) in fixed_edges:
                continue

            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            sep_nodes = possibly_d_sep_sets(pag, i, j, max_path_length=max_path_length)  # type: ignore

            # check that number of adjacencies is greater then the
            # cardinality of the conditioning set
            if len(sep_nodes) >= size_cond_set:
                # loop through all possible conditioning sets of certain size
                for cond_set in combinations(sep_nodes, size_cond_set):
                    # compute conditional independence test
                    _, pvalue = ci_estimator(X, i, j, set(cond_set), **ci_estimator_kwargs)

                    # two variables found to be independent given a separating set
                    if pvalue > alpha:
                        if adj_graph.has_edge(i, j):
                            remove_edges.append((i, j))
                        sep_set[i][j] |= set(cond_set)
                        sep_set[j][i] |= set(cond_set)
                        break
                cont = True
        size_cond_set += 1

        # finally remove edges after performing
        # conditional independence tests
        adj_graph.remove_edges_from(remove_edges)

        # determine if we reached the maximum number of conditioning,
        # or we pruned all possible permutations of nodes
        if size_cond_set > max_cond_set_size or cont is False:
            break

    return adj_graph, sep_set


def learn_skeleton_graph(
    X: pd.DataFrame,
    adj_graph: nx.Graph,
    sep_set: Dict[str, Dict[str, Set]],
    ci_estimator: Callable,
    fixed_edges: Set = None,
    alpha: float = 0.05,
    min_cond_set_size: int = 0,
    max_cond_set_size: int = None,
    **ci_estimator_kwargs,
) -> nx.Graph:
    """Learn a graph from data.

    Proceed by testing neighboring nodes.

    Parameters
    ----------
    X : pandas.DataFrame
        A dataframe consisting of nodes as columns
        and samples as rows.
    adj_graph : networkx.Graph
        The initialized graph. Can be for example a complete graph.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    fixed_edges : set
        The set of fixed edges.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    min_cond_set_size : int
        The minimum size of the conditioning set, by default 0. The number of variables
        used in the conditioning set.
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

    See Also
    --------
    causal_networkx.algorithms.possibly_d_sep_sets
    """
    if max_cond_set_size is None:
        max_cond_set_size = np.inf
    if fixed_edges is None:
        fixed_edges = set()

    nodes = adj_graph.nodes
    size_cond_set = min_cond_set_size

    while 1:
        cont = False
        remove_edges = []

        # loop through all possible permutation of
        # two nodes in the graph
        for (i, j) in permutations(nodes, 2):
            # ignore fixed edges
            if (i, j) in fixed_edges:
                continue

            # determine how we want to construct the candidates for separating nodes
            # check that neighbors for "i" contain "j"
            sep_nodes = list(adj_graph.neighbors(i))
            if j not in sep_nodes:
                continue
            sep_nodes.remove(j)

            # check that number of adjacencies is greater then the
            # cardinality of the conditioning set
            if len(sep_nodes) >= size_cond_set:
                # loop through all possible conditioning sets of certain size
                for cond_set in combinations(sep_nodes, size_cond_set):
                    # compute conditional independence test
                    _, pvalue = ci_estimator(X, i, j, set(cond_set), **ci_estimator_kwargs)

                    # two variables found to be independent given a separating set
                    if pvalue > alpha:
                        if adj_graph.has_edge(i, j):
                            remove_edges.append((i, j))
                        sep_set[i][j] |= set(cond_set)
                        sep_set[j][i] |= set(cond_set)
                        break
                cont = True
        size_cond_set += 1

        # finally remove edges after performing
        # conditional independence tests
        adj_graph.remove_edges_from(remove_edges)

        # determine if we reached the maximum number of conditioning,
        # or we pruned all possible permutations of nodes
        if size_cond_set > max_cond_set_size or cont is False:
            break

    return adj_graph, sep_set
