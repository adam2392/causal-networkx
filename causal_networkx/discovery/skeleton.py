import logging
from collections import defaultdict
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Set, Union

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx import PAG
from causal_networkx.algorithms.pag import pds_path, possibly_d_sep_sets
from causal_networkx.ci import BaseConditionalIndependenceTest

logger = logging.getLogger()

ACCEPTED_MARKOVIAN_SKELETON_METHODS = ["complete", "neighbors", "neighbors_path"]


def _find_neighbors_along_path(G, possible_adjacencies, start, end):
    def _assign_weight(u, v, edge_attr):
        if u == node or v == node:
            return np.inf
        else:
            return 1

    nghbrs = []
    for node in possible_adjacencies:
        if not G.has_edge(start, node):
            raise RuntimeError(f"{start} and {node} are not connected, but they are assumed to be.")

        # find a path from start node to end
        path = nx.shortest_path(G, source=node, target=end, weight=_assign_weight)
        if len(path) > 0:
            if start in path:
                raise RuntimeError("wtf?")
            nghbrs.append(node)
    return nghbrs


class LearnSkeleton:
    """Learn a skeleton graph from data.

    Proceed by testing neighboring nodes, while keeping track of test
    statistic values (these are the ones that are
    the "most dependent"). Remember we are testing the null hypothesis

    .. math::
        H_0: X \\perp Y | Z

    where the alternative hypothesis is that they are dependent and hence
    require a causal edge linking the two variables.

    Parameters
    ----------
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    adj_graph : networkx.Graph, optional
        The initialized graph. Can be for example a complete graph.
        If ``None``, then a complete graph will be initialized.
    sep_set : dictionary of dictionary of list of sets
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
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
    skeleton_method : str
        The method to use for testing conditional independence. Must be one of
        ('complete', 'neighbors', 'neighbors_path'). See Notes for more details.
    keep_sorted : bool
        Whether or not to keep the considered adjacencies in sorted dependency order.
        If True (default) will sort the existing adjacencies of each variable by its
        dependencies from strongest to weakest (i.e. largest CI test statistic value to lowest).
    with_mci : bool
        Whether or not to run the MCI conditioning phase. By default False.
    max_conds_x : int
        If ``with_mci=True``, then this controls the number of conditioning variables
        from the MCI set of variable 'x'.
    max_conds_y : int
        If ``with_mci=True``, then this controls the number of conditioning variables
        from the MCI set of variable 'y'.
    parent_dep_dict : Dict[str, Dict[str, float]]
        The dependency dictionary from variables to their proposed parents.
    size_inclusive : bool
        Whether to include the MCI conditioning set in the ``cond_set_size`` count for
        the overall conditioning set.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph.
    sep_set_ : dictionary of dictionary of list of sets
        Mapping node to other nodes to separating sets of variables.
    test_stat_dict_ : dictionary of dictionary of float
        Mapping the candidate parent-child edge to the smallest absolute value
        test statistic seen in testing 'x' || 'y' given some conditioning set.
    pvalue_dict_ : dictionary of dictionary of float
        Mapping the candidate parent-child edge to the largest pvalue
        seen in testing 'x' || 'y' given some conditioning set.
    stat_min_dict_ : dictionary of dictionary of float
        Mapping the candidate parent-child edge to the smallest test statistic
        seen in testing 'x' || 'y' given some conditioning set.

    See Also
    --------
    causal_networkx.algorithms.possibly_d_sep_sets

    Notes
    -----
    Overview of learning causal skeleton from data:

        This algorithm consists of four general loops through the data.

        - "infinite" loop through size of the conditioning set, 'size_cond_set'
        - loop through nodes of the graph, 'x_var'
        - loop through variables adjacent to selected node, 'y_var'
        - loop through combinations of the conditioning set of size p, 'cond_set'

        At each iteration of the outer infinite loop, the edges for 'size_cond_set'
        are removed and 'size_cond_set' is incremented.

        Furthermore, the maximum pvalue is stored for existing
        dependencies among variables (i.e. any two nodes with an edge still).
        The ``keep_sorted`` hyperparameter keeps the considered parents in
        a sorted order. The ``max_combinations`` parameter allows one to
        limit the fourth loop through combinations of the conditioning set.

        The stopping condition is when the size of the conditioning variables for all (X, Y)
        pairs is less than the size of 'size_cond_set', or if the 'max_cond_set_size' is
        reached.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``skeleton_method`` completely defines how one
        selects the conditioning set.

        - 'complete': This exhaustively conditions on all combinations of variables in
        the graph. This essentially refers to the SGS algorithm :footcite:`Spirtes1993`
        - 'neighbors': This only conditions on adjacent variables to that of 'x_var' and 'y_var'. This
        refers to the traditional PC algorithm :footcite:`Meek1995`
        - 'neighbors_path': This is 'neighbors', but restricts to variables with an adjacency path
        from 'x_var' to 'y_var'. This is a variant from the RFCI paper :footcite:`Colombo2012`
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        adj_graph: nx.Graph = None,
        sep_set: Dict[str, Dict[str, List[Set[Any]]]] = None,
        fixed_edges: Set = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: int = None,
        max_combinations: int = None,
        skeleton_method: str = "neighbors",
        keep_sorted: bool = False,
        with_mci: bool = False,
        max_conds_x: int = None,
        max_conds_y: int = None,
        parent_dep_dict: Dict[str, Dict[str, float]] = None,
        multicomp_method: str = None,
        **ci_estimator_kwargs,
    ) -> None:
        self.ci_estimator = ci_estimator
        self.adj_graph = adj_graph
        self.sep_set = sep_set
        self.fixed_edges = fixed_edges
        self.alpha = alpha
        self.ci_estimator_kwargs = ci_estimator_kwargs
        self.skeleton_method = skeleton_method

        # control of the conditioning set
        self.min_cond_set_size = min_cond_set_size
        self.max_cond_set_size = max_cond_set_size
        self.max_combinations = max_combinations

        # for tracking strength of dependencies
        self.keep_sorted = keep_sorted
        self.parent_dep_dict = parent_dep_dict

        # control of the optional MCI conditioning
        self.with_mci = with_mci
        self.max_conds_x = max_conds_x
        self.max_conds_y = max_conds_y

        self.multicomp_method = multicomp_method

    def _initialize_params(self, nodes):
        """Initialize parameters for learning skeleton.

        Parameters
        ----------
        nodes : list of nodes
            The list of nodes that will be present in the learned skeleton graph.
        """
        # error checks of passed in arguments
        if self.with_mci and self.parent_dep_dict is None:
            raise RuntimeError(
                "Cannot run skeleton discovery with MCI if "
                "parent dependency dictionary (parent_dep_dict) is not passed."
            )
        if self.max_combinations is not None and self.max_combinations <= 0:
            raise RuntimeError(f"Max combinations must be at least 1, not {self.max_combinations}")

        if self.skeleton_method not in ACCEPTED_MARKOVIAN_SKELETON_METHODS:
            raise ValueError(
                f"Skeleton method must be one of {ACCEPTED_MARKOVIAN_SKELETON_METHODS}, not {self.skeleton_method}."
            )

        # error check skeleton method and required arguments for FCI case
        if self.skeleton_method.startswith("pds") and self.pag is None:
            raise RuntimeError(
                'To learn a skeleton using the "pds" method, one must pass in a '
                "PAG to the algorithm."
            )

        # set default values
        if self.adj_graph is None:
            self.adj_graph_ = nx.complete_graph(nodes, create_using=nx.Graph)
        else:
            self.adj_graph_ = self.adj_graph
        if self.fixed_edges is None:
            self.fixed_edges_ = set()
        else:
            self.fixed_edges_ = self.fixed_edges
        if self.sep_set is None:
            # keep track of separating sets
            self.sep_set_ = defaultdict(lambda: defaultdict(list))
        else:
            self.sep_set_ = self.sep_set

        # control of the conditioning set
        if self.max_cond_set_size is None:
            self.max_cond_set_size_ = np.inf
        else:
            self.max_cond_set_size_ = self.max_cond_set_size
        if self.min_cond_set_size is None:
            self.min_cond_set_size_ = 0
        else:
            self.min_cond_set_size_ = self.min_cond_set_size
        if self.max_combinations is None:
            self.max_combinations_ = np.inf
        else:
            self.max_combinations_ = self.max_combinations

        # control of the optional MCI conditioning
        if self.max_conds_x is None:
            self.max_conds_x_ = None
        else:
            self.max_conds_x_ = self.max_conds_x
        if self.max_conds_y is None:
            self.max_conds_y_ = None
        else:
            self.max_conds_y_ = self.max_conds_y

        # parameters to track progress of the algorithm
        self.remove_edges_ = set()

    def fit(self, X: Union[Dict[Any, pd.DataFrame], pd.DataFrame]):
        """Run structure learning to learn the skeleton of the causal graph.

        Parameters
        ----------
        X : pandas.DataFrame
            A dataframe consisting of nodes as columns
            and samples as rows.
        """
        if self.multicomp_method is not None:
            import pingouin as pg

        # perform error-checking and extract node names
        if isinstance(X, dict):
            # the data passed in are instances of multiple distributions
            for idx, (_, X_dataset) in enumerate(X.items()):
                if idx == 0:
                    check_nodes = X_dataset.columns
                nodes = X_dataset.columns
                if not check_nodes.equals(nodes):
                    raise RuntimeError(
                        "All dataset distributions should have the same node names in their columns."
                    )

            # convert final series of nodes to a list
            nodes = nodes.values
        else:
            nodes = X.columns.values

        # initialize learning parameters
        self._initialize_params(nodes)
        adj_graph = self.adj_graph_

        # store the absolute value of test-statistic values for every single
        # candidate parent-child edge (X -> Y)
        self.test_stat_dict_: Dict[Any, Dict[Any, float]] = {
            x_var: {y_var: np.inf for y_var in adj_graph.neighbors(x_var) if y_var != x_var}
            for x_var in nodes
        }

        # store the actual minimum test-statistic/pvalue value for every
        # single candidate parent-child edge
        self.pvalue_dict_: Dict[Any, Dict[Any, float]] = {
            x_var: {y_var: 0 for y_var in adj_graph.neighbors(x_var) if y_var != x_var}
            for x_var in nodes
        }
        self.stat_min_dict_: Dict[Any, Dict[Any, float]] = {
            x_var: {y_var: np.inf for y_var in adj_graph.neighbors(x_var) if y_var != x_var}
            for x_var in nodes
        }

        # store the list of potential adjacencies for every node
        # which is tracked and updated in the algorithm
        adjacency_mapping: Dict[Any, List] = dict()
        for node in nodes:
            adjacency_mapping[node] = [
                other_node for other_node in adj_graph.neighbors(node) if other_node != node
            ]

        logger.info(
            f"\n\nRunning skeleton phase with: \n"
            f"max_combinations: {self.max_combinations_},\n"
            f"min_cond_set_size: {self.min_cond_set_size_},\n"
            f"max_cond_set_size: {self.max_cond_set_size_},\n"
        )
        size_cond_set = self.min_cond_set_size_

        # Outer loop: iterate over 'size_cond_set' until stopping criterion is met
        # - 'size_cond_set' > 'max_cond_set_size' or
        # - All (X, Y) pairs have candidate conditioning sets of size < 'size_cond_set'
        while 1:
            cont = False
            # initialize set of edges to remove at the end of every loop
            self.remove_edges_ = set()

            # loop through every node
            for x_var in adj_graph.nodes:
                # possible_adjacencies = set(adj_graph.neighbors(x_var))
                possible_adjacencies = set(adjacency_mapping[x_var]).copy()

                # keep track of the size of the adjacency set of 'X' without 'Y'
                size_adjacencies_x = len(possible_adjacencies) - 1

                logger.info(f"On node {x_var}\n\n")

                for y_var in possible_adjacencies:
                    # a node cannot be a parent to itself in DAGs
                    if y_var == x_var:
                        continue

                    # ignore fixed edges
                    if (x_var, y_var) in self.fixed_edges_:
                        continue

                    # compute the possible variables used in the conditioning set
                    possible_variables = self._compute_candidate_conditioning_sets(
                        possible_adjacencies,
                        adj_graph,
                        x_var,
                        y_var,
                        skeleton_method=self.skeleton_method,
                    )

                    logger.debug(
                        f"Adj({x_var}) without {y_var} with size={size_adjacencies_x} with p={size_cond_set}. "
                        f"The possible variables to condition on are: {possible_variables}."
                    )

                    # check that number of adjacencies is greater then the
                    # cardinality of the conditioning set
                    if len(possible_variables) < size_cond_set:
                        logger.debug(
                            f"\n\nBreaking for {x_var}, {y_var}, {size_adjacencies_x}, "
                            f"{size_cond_set}, {possible_variables}"
                        )
                        break
                    else:
                        cont = True

                    # get the additional conditioning set if we want to use
                    # the MCI condition
                    if self.with_mci:
                        mci_inclusion_set = self._compute_mci_set(
                            x_var, y_var, parent_dep_dict=self.parent_dep_dict  # type: ignore
                        )
                    else:
                        mci_inclusion_set = set()

                    # generate iterator through the conditioning sets
                    conditioning_sets = _iter_conditioning_set(
                        possible_variables=possible_variables,
                        x_var=x_var,
                        y_var=y_var,
                        size_cond_set=size_cond_set,
                        mci_inclusion_set=mci_inclusion_set,
                    )

                    # now iterate through the possible parents
                    multicomp_pvalues = []
                    for comb_idx, cond_set in enumerate(conditioning_sets):
                        # check the number of combinations of possible parents we have tried
                        # to use as a separating set
                        if (
                            self.max_combinations_ is not None
                            and comb_idx >= self.max_combinations_
                        ):
                            break

                        # compute conditional independence test
                        test_stat, pvalue = self.ci_estimator.test(
                            X, x_var, y_var, set(cond_set), **self.ci_estimator_kwargs
                        )

                        # if we adjust for multicomp locally, then we will store all pvalues until the end
                        if self.multicomp_method is not None:
                            multicomp_pvalues.append(pvalue)
                        elif pvalue > self.alpha:
                            break

                    if self.multicomp_method is not None:
                        _, multicomp_pvalues = pg.multicomp(
                            multicomp_pvalues, alpha=self.alpha, method=self.multicomp_method
                        )

                        # extract one pvalue to use
                        pvalue = np.min(multicomp_pvalues)

                    # check if we would remove the edge
                    removed_edge = pvalue > self.alpha

                    # Notes: positives = links detected, negatives = links removed.
                    # re-run edges using MCI condition, in order to control for false-negatives (removing
                    # edges that shouldn't be removed).
                    if self.with_mci and removed_edge and len(mci_inclusion_set) > 0:
                        pvalue = self._rerun_ci_test_with_mci(
                            X, x_var, y_var, cond_set, mci_inclusion_set
                        )
                        removed_edge = self._postprocess_ci_test(
                            x_var, y_var, cond_set, test_stat, pvalue
                        )
                    else:
                        # post-process the CI test results
                        removed_edge = self._postprocess_ci_test(
                            x_var, y_var, cond_set, test_stat, pvalue
                        )

                    # exit loop if we have found an independency and removed the edge
                    statistic_summary = f"Statistical summary:\n"
                    # f"- ({cond_set}) with MCI={mci_inclusion_set}\n"\
                    # f"- alpha={self.alpha}, pvalue={pvalue}\n"\
                    # f"- size_cond_set={size_cond_set}"
                    if removed_edge:
                        logger.info(f"Removing edge {x_var}, {y_var}... \n{statistic_summary}")
                    else:
                        logger.info(
                            f"Did not remove edge {x_var}, {y_var}... \n{statistic_summary}"
                        )

            # finally remove edges after performing
            # conditional independence tests
            logger.info(f"For p = {size_cond_set}, removing all edges: {self.remove_edges_}")

            # Remove non-significant links from the test statistic and pvalue dict
            for x_var, y_var in self.remove_edges_:
                self.test_stat_dict_[x_var].pop(y_var, None)
                self.stat_min_dict_[x_var].pop(y_var, None)
                self.pvalue_dict_[x_var].pop(y_var, None)

                self.test_stat_dict_[y_var].pop(x_var, None)
                self.stat_min_dict_[y_var].pop(x_var, None)
                self.pvalue_dict_[y_var].pop(x_var, None)

            adj_graph.remove_edges_from(self.remove_edges_)

            # also remove them from the parent dict mapping
            for x_var in self.test_stat_dict_.keys():
                # variable mapping to its adjacencies and absolute value of their current dependencies
                # assuming there is still an edge (if the pvalue rejected the null hypothesis)
                abs_values = {
                    k: np.abs(self.test_stat_dict_[x_var][k])
                    for k in self.test_stat_dict_[x_var].keys()
                }

                if self.keep_sorted:
                    # sort the parents and re-assign possible parents based on this
                    # ordering, which is used in the next loop for a conditioning set size.
                    # Pvalues are sorted in ascending order, so that means most dependent to least dependent
                    # Therefore test statistic values are sorted in descending order.
                    possible_adjacencies_ = sorted(abs_values, key=abs_values.get, reverse=True)  # type: ignore
                else:
                    # logger.debug(f"{node} - {possible_adjacencies}, {list(abs_values.keys())}")
                    possible_adjacencies_ = list(abs_values.keys())

                adjacency_mapping[x_var] = possible_adjacencies_

            # increment the conditioning set size
            size_cond_set += 1

            # only allow conditioning set sizes up to maximum set number
            if size_cond_set > self.max_cond_set_size_ or cont is False:
                break

        self.adj_graph_ = adj_graph

    def _compute_candidate_conditioning_sets(
        self, possible_adjacencies, adj_graph: nx.Graph, x_var, y_var, skeleton_method: str
    ) -> Set[Any]:
        """Compute candidate conditioning sets.

        Parameters
        ----------
        possible_adjacencies : Dict of Dict of OrderedSet
            Possible adjacencies for each ('x_var', 'y_var') that is possibly ordered
            based on dependency.
        adj_graph : nx.Graph
            The adjacency graph.
        x_var : node
            The 'X' node.
        y_var : node
            The 'Y' node.
        skeleton_method : str
            The skeleton method, which dictates how we choose the corresponding
            conditioning sets.

        Returns
        -------
        possible_variables : Set
            The set of nodes in 'adj_graph' that are candidates for the
            conditioning set.
        """
        if skeleton_method == "complete":
            possible_variables = adj_graph.nodes
        elif skeleton_method == "neighbors":
            possible_variables = possible_adjacencies.copy()
        elif skeleton_method == "neighbors_path":
            # constrain adjacency set to ones with a path from x_var to y_var
            possible_variables = _find_neighbors_along_path(
                adj_graph, possible_adjacencies, start=x_var, end=y_var
            )

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def _compute_mci_set(
        self, x_var, y_var, parent_dep_dict: Dict[str, Dict[str, float]]
    ) -> Set[Any]:
        """Compute the MCI conditioning set.

        Parameters
        ----------
        x_var : node
            The node name for 'x'.
        y_var : node
            The node name for 'y'.
        parent_dep_dict : Dict[str, Dict[str, float]]
            The dependency dictionary from variables to their proposed parents.

        Returns
        -------
        mci_set : Set[Any]
            The MCI conditioning set that does not include 'x', or 'y'.
        """
        mci_set: Set[Any]

        # get the additional conditioning set if MCI
        possible_conds_x = list(parent_dep_dict[x_var].keys())  # type: ignore
        conds_x = set(possible_conds_x[: self.max_conds_x_])
        possible_conds_y = list(parent_dep_dict[y_var].keys())  # type: ignore
        conds_y = set(possible_conds_y[: self.max_conds_y_])

        # make sure X and Y are not in the additional conditionals
        mci_set = conds_x.union(conds_y)

        if x_var in mci_set:
            mci_set.remove(x_var)
        if y_var in mci_set:
            mci_set.remove(y_var)
        return mci_set

    def _postprocess_ci_test(self, x_var, y_var, cond_set, test_stat, pvalue):
        # keep track of the smallest test statistic, meaning the highest pvalue
        # meaning the "most" independent
        if np.abs(test_stat) <= self.test_stat_dict_[x_var].get(y_var, np.inf):
            logger.debug(f"Adding {y_var} to possible adjacency of node {x_var}")
            self.test_stat_dict_[x_var][y_var] = np.abs(test_stat)

        # keep track of the maximum pvalue as well
        if pvalue > self.pvalue_dict_[x_var].get(y_var, -0.1):
            self.pvalue_dict_[x_var][y_var] = pvalue
            self.stat_min_dict_[x_var][y_var] = test_stat

        # two variables found to be independent given a separating set
        if pvalue > self.alpha:
            self.remove_edges_.add((x_var, y_var))
            self.sep_set_[x_var][y_var].append(set(cond_set))
            self.sep_set_[y_var][x_var].append(set(cond_set))
            return True
        return False

    def _rerun_ci_test_with_mci(self, X, x_var, y_var, cond_set, mci_inclusion_set) -> float:
        pvalues = []
        # iterate through the MCI set adding it incrementally to 'Z' to
        # test the hypothesis that the removal of an edge was correct
        mci_size = len(mci_inclusion_set)
        for p_size in range(1, mci_size):
            for mci_set in combinations(mci_inclusion_set, p_size):
                aug_cond_set = set(cond_set).copy().union(mci_set)

                # re-run the test with augmented conditioning set
                _, pvalue = self.ci_estimator.test(
                    X, x_var, y_var, aug_cond_set, **self.ci_estimator_kwargs
                )
                if pvalue < 0.05:
                    return pvalue
                pvalues.append(pvalue)
        return np.amin(pvalues)


class LearnSemiMarkovianSkeleton(LearnSkeleton):
    """Learn a skeleton graph from data and a partial skeleton.

    Parameters
    ----------
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    adj_graph : networkx.Graph, optional
        The initialized graph. Can be for example a complete graph.
        If ``None``, then a complete graph will be initialized.
    sep_set : dictionary of dictionary of list of sets
        Mapping node to other nodes to separating sets of variables.
        If ``None``, then an empty dictionary of dictionary of list of sets
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
    skeleton_method : str
        The method to use for testing conditional independence. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    max_path_length : int
        The maximum length of a path to consider when looking for possibly d-separating
        sets among two nodes. Only used if ``skeleton_method=pds``. Default is infinite.
    pag : PAG
        The partial ancestral graph. Only used if ``skeleton_method=pds``.
    keep_sorted : bool
        Whether or not to keep the considered adjacencies in sorted dependency order.
        If True (default) will sort the existing adjacencies of each variable by its
        dependencies from strongest to weakest (i.e. largest CI test statistic value to lowest).
    with_mci : bool
        Whether or not to run the MCI conditioning phase. By default False.
    max_conds_x : int
        If ``with_mci=True``, then this controls the number of conditioning variables
        from the MCI set of variable 'x'.
    max_conds_y : int
        If ``with_mci=True``, then this controls the number of conditioning variables
        from the MCI set of variable 'y'.
    parent_dep_dict : Dict[str, Dict[str, float]]
        The dependency dictionary from variables to their proposed parents.
    size_inclusive : bool
        Whether to include the MCI conditioning set in the ``cond_set_size`` count for
        the overall conditioning set.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    adj_graph_ : nx.Graph
        The discovered graph from data. Stored using an undirected
        graph.
    sep_set_ : dictionary of dictionary of list of sets
        Mapping node to other nodes to separating sets of variables.
    test_stat_dict_ : dictionary of dictionary of float
        Mapping the candidate parent-child edge to the smallest absolute value
        test statistic seen in testing 'x' || 'y' given some conditioning set.
    pvalue_dict_ : dictionary of dictionary of float
        Mapping the candidate parent-child edge to the largest pvalue
        seen in testing 'x' || 'y' given some conditioning set.
    stat_min_dict_ : dictionary of dictionary of float
        Mapping the candidate parent-child edge to the smallest test statistic
        seen in testing 'x' || 'y' given some conditioning set.

    See Also
    --------
    causal_networkx.algorithms.possibly_d_sep_sets

    Notes
    -----
    Overview of learning causal skeleton from data:

        This algorithm consists of four general loops through the data.

        - "infinite" loop through size of the conditioning set, 'size_cond_set'
        - loop through nodes of the graph, 'x_var'
        - loop through variables adjacent to selected node, 'y_var'
        - loop through combinations of the conditioning set of size p, 'cond_set'

        At each iteration of the outer infinite loop, the edges for 'size_cond_set'
        are removed and 'size_cond_set' is incremented.

        Furthermore, the maximum pvalue is stored for existing
        dependencies among variables (i.e. any two nodes with an edge still).
        The ``keep_sorted`` hyperparameter keeps the considered parents in
        a sorted order. The ``max_combinations`` parameter allows one to
        limit the fourth loop through combinations of the conditioning set.

        The stopping condition is when the size of the conditioning variables for all (X, Y)
        pairs is less than the size of 'size_cond_set', or if the 'max_cond_set_size' is
        reached.

    Different methods for learning the skeleton:

        There are different ways to learn the skeleton that are valid under various
        assumptions. The value of ``skeleton_method`` completely defines how one
        selects the conditioning set.

        - 'pds': This constructs the set of possibly d-separating nodes for a variable 'x_var', which
        is useful in FCI-variant algorithms :footcite:`Colombo2012`.
        - 'pds_path': This is 'pds', but restricts to variables that are in the biconnected components
        of the adjacency graph that contains the edge (x_var, y_var) :footcite:`Colombo2012`.
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        adj_graph: nx.Graph = None,
        sep_set: Dict[str, Dict[str, List[Set[Any]]]] = None,
        fixed_edges: Set = None,
        alpha: float = 0.05,
        min_cond_set_size: int = 0,
        max_cond_set_size: int = None,
        max_combinations: int = None,
        skeleton_method: str = "neighbors",
        max_path_length: int = np.inf,
        pag: PAG = None,
        keep_sorted: bool = False,
        with_mci: bool = False,
        max_conds_x: int = None,
        max_conds_y: int = None,
        parent_dep_dict: Dict[str, Dict[str, float]] = None,
        multicomp_method: str = None,
        **ci_estimator_kwargs,
    ) -> None:
        super().__init__(
            ci_estimator,
            adj_graph,
            sep_set,
            fixed_edges,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            keep_sorted,
            with_mci,
            max_conds_x,
            max_conds_y,
            parent_dep_dict,
            multicomp_method,
            **ci_estimator_kwargs,
        )

        # for controlling paths over estimated graph
        self.pag = pag
        self.max_path_length = max_path_length

    def _compute_candidate_conditioning_sets(
        self, possible_adjacencies, adj_graph: nx.Graph, x_var, y_var, skeleton_method: str
    ) -> Set[Any]:
        """Compute candidate conditioning sets.

        Parameters
        ----------
        possible_adjacencies : Dict of Dict of OrderedSet
            Possible adjacencies for each ('x_var', 'y_var') that is possibly ordered
            based on dependency.
        adj_graph : nx.Graph
            The adjacency graph.
        x_var : node
            The 'X' node.
        y_var : node
            The 'Y' node.
        skeleton_method : str
            The skeleton method, which dictates how we choose the corresponding
            conditioning sets.

        Returns
        -------
        possible_variables : Set
            The set of nodes in 'adj_graph' that are candidates for the
            conditioning set.
        """
        if skeleton_method == "pds":
            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            possible_variables = possibly_d_sep_sets(
                self.pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
            )
            if (x_var == "A" and y_var == "E") or (y_var == "A" and x_var == "E"):
                print(possible_variables)
        elif skeleton_method == "pds_path":
            # determine how we want to construct the candidates for separating nodes
            # perform conditioning independence testing on all combinations
            possible_variables = pds_path(
                self.pag, x_var, y_var, max_path_length=self.max_path_length  # type: ignore
            )

        if x_var in possible_variables:
            possible_variables.remove(x_var)
        if y_var in possible_variables:
            possible_variables.remove(y_var)

        return possible_variables

    def fit(self, X: Union[Dict[Any, pd.DataFrame], pd.DataFrame]):
        return super().fit(X)


def _rerun_ci_tests_with_mci(
    data, x_var, y_var, z_covariates, ci_estimator, mci_set, size_mci, **ci_estimator_kwargs
):
    z_covariates = set(z_covariates)

    # keep track of all pvalues
    pvalues = []
    test_stats = []
    for cond_aug_set in combinations(mci_set, size_mci):
        # augment the conditioning set
        cond_set = z_covariates.union(cond_aug_set)

        # compute conditional independence test
        test_stat, pvalue = ci_estimator.test(data, x_var, y_var, cond_set, **ci_estimator_kwargs)
        pvalues.append(pvalue)
        test_stats.append(test_stat)

    return test_stats, pvalues


def _iter_conditioning_set(
    possible_variables: Iterable,
    x_var,
    y_var,
    size_cond_set: int,
    mci_inclusion_set: Set = set(),
    size_inclusive: bool = True,
):
    """Iterate function to generate the conditioning set.

    Parameters
    ----------
    possible_variables : iterable
        A set/list/dict of possible variables to consider for the conditioning set.
        This can be for example, the current adjacencies.
    x_var : node
        The node for the 'x' variable.
    y_var : node
        The node for the 'y' variable.
    size_cond_set : int
        The size of the conditioning set to consider. If there are
        less adjacent variables than this number, then all variables will be in the
        conditioning set.
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
    exclusion_set = {x_var, y_var}

    all_adj_excl_current = [
        p for p in possible_variables if p not in exclusion_set and p not in mci_inclusion_set
    ]

    # set the conditioning size to be the passed in size minus the MCI set if we are inclusive
    # else, set it to the passed in size
    cond_size = size_cond_set - len(mci_inclusion_set) if size_inclusive else size_cond_set

    # if size_inclusive and mci set is larger, then we will just return the MCI set
    if cond_size < 0:
        return [mci_inclusion_set]

    # loop through all possible combinations of the conditioning set size
    for cond in combinations(all_adj_excl_current, cond_size):
        cond_set = set(cond).union(mci_inclusion_set)
        yield cond_set


# TODO: refactor into the class
def learn_skeleton_graph_with_pdsep(
    X: pd.DataFrame,
    ci_estimator: BaseConditionalIndependenceTest,
    adj_graph: nx.Graph = None,
    sep_set: Dict[str, Dict[str, List[Set[Any]]]] = None,
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
        sep_set = defaultdict(lambda: defaultdict(list))
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
                    _, pvalue = ci_estimator.test(X, i, j, set(cond_set), **ci_estimator_kwargs)
                    # two variables found to be independent given a separating set
                    if pvalue > alpha:
                        if adj_graph.has_edge(i, j):
                            remove_edges.append((i, j))
                        sep_set[i][j].append(set(cond_set))
                        sep_set[j][i].append(set(cond_set))
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
