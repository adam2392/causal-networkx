import logging
from collections import defaultdict
from itertools import combinations, permutations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx import PAG
from causal_networkx.algorithms.pag import possibly_d_sep_sets

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
                        logger.info(
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
    keep_sorted: bool = True,
    with_mci: bool = False,
    max_conds_x: int = None,
    max_conds_y: int = None,
    parent_dep_dict: Dict[str, Dict[str, float]] = None,
    size_inclusive: bool = False,
    only_mci: bool = False,
    **ci_estimator_kwargs,
) -> Tuple[
    nx.Graph, Dict[str, Dict[str, Set]], Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, float]]
]:
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
    keep_sorted : bool
        Whether or not to keep the considered adjacencies in sorted dependency order.
        If True (default) will sort the existing adjacencies of each variable by its
        dependencies from strongest to weakest (i.e. largest CI test statistic value to lowest).
    with_mci : bool
        False.
    max_conds_x : int

    max_conds_y : int

    parent_dep_dict : Dict[str, Dict[str, float]]

    size_inclusive : bool

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

    Notes
    -----
    This algorithm consists of four loops through the data:

    - loop through nodes of the graph
    - loop through size of the conditioning set, p
    - loop through current adjacencies
    - loop through combinations of the conditioning set of size p

    At each iteration, the maximum pvalue is stored for existing
    dependencies among variables (i.e. any two nodes with an edge still).
    The ``keep_sorted`` hyperparameter keeps the considered parents in
    a sorted order. The ``max_combinations`` parameter allows one to
    limit the fourth loop through combinations of the conditioning set.

    The iteration through combination of the conditioning set only
    considers adjacencies of the existing variables.
    """
    # error checks of passed in arguments
    if with_mci and parent_dep_dict is None:
        raise RuntimeError(
            "Cannot run skeleton discovery with MCI if "
            "parent dependency dictionary (parent_dep_dict) is not passed."
        )
    if max_combinations is not None and max_combinations <= 0:
        raise RuntimeError(f"Max combinations must be at least 1, not {max_combinations}")

    # set default values
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

    # store the absolute value of test-statistic values for every single
    # candidate parent-child edge (X -> Y)
    test_stat_dict: Dict[Any, Dict[Any, float]] = dict()
    pvalue_dict: Dict[Any, Dict[Any, float]] = dict()

    # store the actual minimum test-statistic value for every
    # single candidate parent-child edge
    stat_min_dict: Dict[Any, Dict[Any, float]] = dict()

    # store the list of potential adjacencies for every node
    # which is tracked and updated in the algorithm
    adjacency_mapping: Dict[Any, List] = dict()
    nodes = adj_graph.nodes
    for node in nodes:
        adjacency_mapping[node] = [
            other_node for other_node in adj_graph.neighbors(node) if other_node != node
        ]

    mci_set: Union[Set[Any], str]

    # loop through every node
    for i in nodes:
        possible_adjacencies = adjacency_mapping[i]

        test_stat_dict[i] = dict()
        pvalue_dict[i] = dict()
        stat_min_dict[i] = dict()

        # get the additional conditioning set if MCI
        if with_mci:
            possible_conds_x = list(parent_dep_dict[i].keys())  # type: ignore
            conds_x = set(possible_conds_x[:max_conds_x])

        # the total number of conditioning set variables
        total_num_vars = len(possible_adjacencies)
        for size_cond_set in range(min_cond_set_size, total_num_vars):
            remove_edges = []

            # if the number of adjacencies is the size of the conditioning set
            # exit the loop and increase the size of the conditioning set
            if len(possible_adjacencies) - 1 < size_cond_set:
                break

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > max_cond_set_size:
                break

            for j in possible_adjacencies:
                # a node cannot be a parent to itself in DAGs
                if j == i:
                    continue

                # ignore fixed edges
                if (i, j) in fixed_edges:
                    continue

                # get the additional conditioning set if MCI
                if with_mci:
                    possible_conds_y = list(parent_dep_dict[j].keys())  # type: ignore
                    conds_y = set(possible_conds_y[:max_conds_y])

                    # make sure X and Y are not in the additional conditionals
                    mci_inclusion_set = conds_x.union(conds_y)

                    if i in mci_inclusion_set:
                        mci_inclusion_set.remove(i)
                    if j in mci_inclusion_set:
                        mci_inclusion_set.remove(j)
                else:
                    mci_inclusion_set = set()

                if not only_mci:
                    conditioning_sets = _iter_conditioning_set(
                        possible_adjacencies,
                        size_cond_set,
                        exclude_var=j,
                        mci_inclusion_set=mci_inclusion_set,
                        size_inclusive=size_inclusive,
                    )
                else:
                    conditioning_sets = combinations(mci_inclusion_set, size_cond_set)

                # now iterate through the possible parents
                # f(possible_adjacencies, size_cond_set, j)
                for comb_idx, cond_set in enumerate(conditioning_sets):
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
                    if with_mci:
                        mci_set = mci_inclusion_set
                        logger.info(f"{conds_x}, {conds_y}")
                    else:
                        mci_set = "No MCI"
                    if pvalue > alpha:
                        logger.info(
                            f"Removing edge {i}-{j} conditioned on {cond_set}: "
                            f"MCI={mci_set}, "
                            f"alpha={alpha}, pvalue={pvalue}"
                        )
                        remove_edges.append((i, j))
                        sep_set[i][j] |= set(cond_set)
                        sep_set[j][i] |= set(cond_set)
                        break
                    else:
                        logger.info(
                            f"Did not remove edge {i}-{j} conditioned on {cond_set}: "
                            f"MCI={mci_set}, "
                            f"alpha={alpha}, pvalue={pvalue}"
                        )

            # finally remove edges after performing
            # conditional independence tests
            logger.info(f"Removed all edges with p = {size_cond_set}")
            adj_graph.remove_edges_from(remove_edges)

            # also remove them from the parent dict mapping
            adjacency_mapping[node] = list(adj_graph.neighbors(node))

            # Remove non-significant links from the test statistic and pvalue dict
            for _, parent in remove_edges:
                test_stat_dict[i].pop(parent)
                pvalue_dict[i].pop(parent)

            # variable mapping to its adjacencies and absolute value of their current dependencies
            # assuming there is still an edge (if the pvalue rejected the null hypothesis)
            abs_values = {k: np.abs(test_stat_dict[i][k]) for k in list(test_stat_dict[i])}

            if keep_sorted:
                # sort the parents and re-assign possible parents based on this
                # ordering, which is used in the next loop for a conditioning set size.
                # Pvalues are sorted in ascending order, so that means most dependent to least dependent
                # Therefore test statistic values are sorted in descending order.
                possible_adjacencies = sorted(abs_values, key=abs_values.get, reverse=True)  # type: ignore
            else:
                possible_adjacencies = list(abs_values.keys())

    return adj_graph, sep_set, test_stat_dict, pvalue_dict


def _iter_conditioning_set(
    possible_adjacencies,
    size_cond_set,
    exclude_var,
    mci_inclusion_set={},
    size_inclusive: bool = True,
):
    """Iterate function to generate the conditioning set.

    Parameters
    ----------
    possible_adjacencies : dict
        A dictionary of possible adjacencies.
    size_cond_set : int
        The size of the conditioning set to consider. If there are
        less adjacent variables than this number, then all variables will be in the
        conditioning set.
    exclude_var : set
        The set of variables to exclude from conditioning.
    mci_inclusion_set : set, optional
        Definite set of variables to include for conditioning, by default None.
    size_incluseive : bool
        Whether or not to include the MCI inclusion set in the count.
        If True (default), then ``mci_inclusion_set`` will be included
        in the ``size_cond_set``. Only if ``size_cond_set`` is greater
        than the size of the ``mci_inclusion_set`` will other possible
        adjacencies be considered.

    Yields
    ------
    Z : set
        The set of variables for the conditioning set.
    """
    all_adj_excl_current = [
        p for p in possible_adjacencies if p != exclude_var and p not in mci_inclusion_set
    ]

    # set the conditioning size to be the passed in size minus the MCI set if we are inclusive
    # else, set it to the passed in size
    cond_size = size_cond_set - len(mci_inclusion_set) if size_inclusive else size_cond_set

    # if size_inclusive and mci set is larger, then we will just return the MCI set
    if cond_size < 0:
        return [mci_inclusion_set]

    # loop through all possible combinations of the conditioning set size
    for cond in combinations(all_adj_excl_current, cond_size):
        cond = set(cond).union(mci_inclusion_set)
        yield cond
