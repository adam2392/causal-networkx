import logging
from collections import defaultdict
from itertools import combinations, permutations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx.algorithms.pag import possibly_d_sep_sets
from causal_networkx.cgm import PAG

logger = logging.getLogger()


def learn_skeleton_graph_with_pdsep(
    X: pd.DataFrame,
    ci_estimator: Callable,
    adj_graph: Optional[nx.Graph] = None,
    sep_set: Optional[Dict[str, Dict[str, Set]]] = None,
    fixed_edges: Optional[Set] = None,
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
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    adj_graph : nx.Graph
        The initialized graph. Can be for example a complete graph.
        If ``None``, then a complete graph will be initialized.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of sets
        will be initialized.
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
    max_path_length : int
        The maximum length of a path to consider when looking for possibly d-separating
        sets among two nodes. Only used if ``only_neighbors=False``. Default is infinite.
    pag : PAG
        The partial ancestral graph. Only used if ``only_neighbors=False``.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Returns
    -------
    adj_graph : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.

    See Also
    --------
    causal_networkx.algorithms.possibly_d_sep_sets
    """
    if adj_graph is None:
        nodes = X.columns
        adj_graph = nx.complete_graph(nodes, create_using=nx.Graph)
    if sep_set is None:
        # keep track of separating sets
        sep_set = defaultdict(lambda: defaultdict(set))

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
            if (i, j) in fixed_edges:  # type: ignore
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


def learn_skeleton_graph_with_neighbors(
    X: pd.DataFrame,
    ci_estimator: Callable,
    adj_graph: Optional[nx.Graph] = None,
    sep_set: Optional[Dict[str, Dict[str, Set]]] = None,
    fixed_edges: Set = None,
    alpha: float = 0.05,
    min_cond_set_size: int = 0,
    max_cond_set_size: int = None,
    **ci_estimator_kwargs,
) -> Tuple[nx.Graph, Dict[str, Dict[str, Set]]]:
    """Learn a skeleton graph from data.

    Proceed by testing neighboring nodes, while keeping track of test
    statistic values (these are the ones that are
    the "most dependent"). Remember we are testing the null hypothesis

    .. math::
        H_0:\ X \\perp Y | Z

    where the alternative hypothesis is that they are dependent and hence
    require a causal edge linking the two variables.

    Parameters
    ----------
    X : pandas.DataFrame
        A dataframe consisting of nodes as columns
        and samples as rows.
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    adj_graph : networkx.Graph, optional
        The initialized graph. Can be for example a complete graph.
        If ``None``, then a complete graph will be initialized.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of sets
        will be initialized.
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
    if adj_graph is None:
        nodes = X.columns
        adj_graph = nx.complete_graph(nodes, create_using=nx.Graph)
    if sep_set is None:
        # keep track of separating sets
        sep_set = defaultdict(lambda: defaultdict(set))

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
                        logger.debug(
                            f"Removing edge {i} {j} with conditioning set {cond_set}: "
                            f"alpha={alpha}, pvalue={pvalue}"
                        )
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


def learn_skeleton_graph_with_order(
    X: pd.DataFrame,
    ci_estimator: Callable,
    adj_graph: nx.Graph = None,
    sep_set: Dict[str, Dict[str, Set]] = None,
    fixed_edges: Set = None,
    alpha: float = 0.05,
    min_cond_set_size: int = 0,
    max_cond_set_size: int = None,
    max_combinations: int = None,
    **ci_estimator_kwargs,
) -> Tuple[nx.Graph, Dict[str, Dict[str, Set]]]:
    """Learn a skeleton graph from data.

    Proceed by testing neighboring nodes, while keeping track of test
    statistic values (these are the ones that are
    the "most dependent"). Remember we are testing the null hypothesis

    .. math::
        H_0:\ X \\perp Y | Z

    where the alternative hypothesis is that they are dependent and hence
    require a causal edge linking the two variables.

    Parameters
    ----------
    X : pandas.DataFrame
        A dataframe consisting of nodes as columns
        and samples as rows.
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    adj_graph : networkx.Graph, optional
        The initialized graph. Can be for example a complete graph.
        If ``None``, then a complete graph will be initialized.
    sep_set : dictionary of dictionary of sets
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of sets
        will be initialized.
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
    max_combinations : int,optional
        Maximum number of tries with a conditioning set chosen from the set of possible
        parents still, by default None. If None, then will not be used. If set, then
        the conditioning set will be chosen lexographically based on the sorted
        test statistic values of 'ith Pa(X) -> X', for each possible parent node of 'X'.
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
    if adj_graph is None:
        nodes = X.columns
        adj_graph = nx.complete_graph(nodes, create_using=nx.Graph)
    if sep_set is None:
        # keep track of separating sets
        sep_set = defaultdict(lambda: defaultdict(set))

    if max_cond_set_size is None:
        max_cond_set_size = np.inf
    if fixed_edges is None:
        fixed_edges = set()
    if max_combinations is not None and max_combinations <= 0:
        raise RuntimeError(f"Max combinations must be at least 1, not {max_combinations}")

    # store the absolute value of test-statistic values for every single
    # candidate parent-child edge (X -> Y)
    test_stat_dict: Dict[Any, Dict[Any, float]] = dict()
    pvalue_dict: Dict[Any, Dict[Any, float]] = dict()

    # store the actual minimum test-statistic value for every
    # single candidate parent-child edge
    stat_min_dict: Dict[Any, Dict[Any, float]] = dict()

    nodes = adj_graph.nodes
    parents_mapping: Dict[Any, List] = dict()
    for node in nodes:
        parents_mapping[node] = [
            other_node for other_node in adj_graph.neighbors(node) if other_node != node
        ]

    # loop through every node
    for i in nodes:
        possible_parents = parents_mapping[i]

        test_stat_dict[i] = dict()
        pvalue_dict[i] = dict()
        stat_min_dict[i] = dict()

        # the total number of conditioning set variables
        total_num_vars = len(possible_parents)
        for size_cond_set in range(min_cond_set_size, total_num_vars):
            remove_edges = []

            # if the number of adjacencies is the size of the conditioning set
            # exit the loop and increase the size of the conditioning set
            if len(possible_parents) - 1 < size_cond_set:
                break

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > max_cond_set_size:
                break

            for j in possible_parents:
                # a node cannot be a parent to itself in DAGs
                if j == i:
                    continue

                # ignore fixed edges
                if (i, j) in fixed_edges:
                    continue

                # now iterate through the possible parents
                # f(possible_parents, size_cond_set, j)
                for comb_idx, cond_set in enumerate(
                    _iter_conditioning_set(possible_parents, size_cond_set, exclude_var=j)
                ):
                    # check the number of combinations of possible parents we have tried
                    # to use as a separating set
                    if max_combinations is not None and comb_idx >= max_combinations:
                        break

                    # compute conditional independence test
                    test_stat, pvalue = ci_estimator(X, i, j, set(cond_set), **ci_estimator_kwargs)

                    # keep track of the smallest test statistic, meaning the highest pvalue
                    # meaning the "most" independent
                    if np.abs(test_stat) < test_stat_dict[i].get(j, np.inf):
                        test_stat_dict[i][j] = np.abs(test_stat)

                    # keep track of the maximum pvalue as well
                    if pvalue > pvalue_dict[i].get(j, 0.0):
                        pvalue_dict[i][j] = pvalue
                        stat_min_dict[i][j] = test_stat

                    # two variables found to be independent given a separating set
                    if pvalue > alpha:
                        remove_edges.append((i, j))
                        sep_set[i][j] |= set(cond_set)
                        sep_set[j][i] |= set(cond_set)
                        break

            # finally remove edges after performing
            # conditional independence tests
            adj_graph.remove_edges_from(remove_edges)

            # also remove them from the parent dict mapping
            parents_mapping[node] = list(adj_graph.neighbors(node))

            # Remove non-significant links
            for _, parent in remove_edges:
                del test_stat_dict[i][parent]

            # sort the parents and re-assign possible parents based on this
            # ordering, which is used in the next loop for a conditioning set size.
            # Pvalues are sorted in ascending order, so that means most dependent to least dependent
            # Therefore test statistic values are sorted in descending order.
            abs_values = {k: np.abs(test_stat_dict[i][k]) for k in list(test_stat_dict[i])}
            possible_parents = sorted(abs_values, key=abs_values.get, reverse=True)  # type: ignore

    return adj_graph, sep_set


def _iter_conditioning_set(possible_parents, size_cond_set, exclude_var):
    all_parents_excl_current = [p for p in possible_parents if p != exclude_var]
    for cond in combinations(all_parents_excl_current, size_cond_set):
        yield list(cond)
