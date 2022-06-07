import itertools
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx import ADMG


# TODO: Add ways to fix directed edges
# TODO: Add ways to initialize graph with edges rather then undirected
class ConstraintDiscovery:
    """Constraint-based algorithms for causal discovery.

    Contains common methods used for all constraint-based causal discovery algorithms.

    Parameters
    ----------
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    init_graph : nx.Graph | ADMG, optional
        An initialized graph. If ``None``, then will initialize PC using a
        complete graph. By default None.
    fixed_edges : nx.Graph, optional
        An undirected graph with fixed edges. If ``None``, then will initialize PC using a
        complete graph. By default None.
    min_cond_set_size : int, optional
        Minimum size of the conditioning set, by default None, which will be set to '0'.
        Used to constrain the computation spent on the algorithm.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
    max_combinations : int, optional
        Maximum number of tries with a conditioning set chosen from the set of possible
        parents still, by default None. If None, then will not be used. If set, then
        the conditioning set will be chosen lexographically based on the sorted
        test statistic values of 'ith Pa(X) -> X', for each possible parent node of 'X'.
    apply_orientations : bool
        Whether or not to apply orientation rules given the learned skeleton graph
        and separating set per pair of variables. If ``True`` (default), will
        apply orientation rules for specific algorithm.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.

    Attributes
    ----------
    graph_ : PAG
        The graph discovered.
    separating_sets_ : dict
        The dictionary of separating sets, where it is a nested dictionary from
        the variable name to the variable it is being compared to the set of
        variables in the graph that separate the two.
    """

    graph_: Optional[Any]
    separating_sets_: Optional[Dict[str, Dict[str, Set[Any]]]]

    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_combinations: int = None,
        apply_orientations: bool = True,
        **ci_estimator_kwargs,
    ):
        self.alpha = alpha
        self.ci_estimator = ci_estimator
        self.ci_estimator_kwargs = ci_estimator_kwargs
        self.init_graph = init_graph
        self.fixed_edges = fixed_edges
        self.apply_orientations = apply_orientations

        if max_cond_set_size is None:
            max_cond_set_size = np.inf
        self.max_cond_set_size = max_cond_set_size
        if min_cond_set_size is None:
            min_cond_set_size = 0
        self.min_cond_set_size = min_cond_set_size
        if max_combinations is None:
            max_combinations = np.inf
        self.max_combinations = max_combinations
        self.separating_sets_ = None
        self.graph_ = None

    def _initialize_graph(self, X: pd.DataFrame):
        nodes = X.columns.values

        # keep track of separating sets
        sep_set: Dict[str, Dict[str, Set]] = defaultdict(lambda: defaultdict(set))

        # initialize the starting graph
        if self.init_graph is None:
            graph = nx.complete_graph(nodes, create_using=nx.Graph)
        else:
            graph = self.init_graph

            if graph.nodes != nodes:
                raise ValueError(
                    f"The nodes within the initial graph, {graph.nodes}, "
                    f"do not match the nodes in the passed in data, {nodes}."
                )

            # since we are not starting from a complete graph,
            # find the separating sets
            for (node_i, node_j) in itertools.combinations(*graph.nodes):
                if not graph.has_edge(node_i, node_j):
                    sep_set[node_i][node_j] = set()
                    sep_set[node_j][node_i] = set()

        # check on fixed edges and keep track
        fixed_edges = set()
        if self.fixed_edges is not None:
            if not np.array_equal(self.fixed_edges.nodes, nodes):
                raise ValueError(
                    f"The nodes within the fixed-edges graph, {self.fixed_edges.nodes}, "
                    f"do not match the nodes in the passed in data, {nodes}."
                )

            for (i, j) in self.fixed_edges.edges:
                fixed_edges.add((i, j))
                fixed_edges.add((j, i))
        return graph, sep_set, fixed_edges

    def orient_edges(self, graph: Any, sep_set: Dict[str, Dict[str, Set]]) -> Any:
        raise NotImplementedError(
            "All constraint discovery algorithms need to implement a function to orient the "
            "skeleton graph given a separating set."
        )

    def _orient_colliders(self, graph: Any, sep_set: Dict[str, Dict[str, Set]]):
        raise NotImplementedError()

    def convert_skeleton_graph(self, graph: nx.Graph) -> Any:
        raise NotImplementedError(
            "All constraint discovery algorithms need to implement a function to convert "
            "the skeleton graph to a causal graph."
        )

    def fit(self, X: pd.DataFrame) -> None:
        """Fit algorithm on dataset 'X'."""
        # initialize graph
        graph, sep_set, fixed_edges = self._initialize_graph(X)

        # learn skeleton graph and the separating sets per variable
        graph, sep_set, _, _ = self.learn_skeleton(X, graph, sep_set, fixed_edges)

        # convert networkx.Graph to relevant causal graph object
        graph = self.convert_skeleton_graph(graph)

        # orient edges on the causal graph object
        if self.apply_orientations:
            graph = self.orient_edges(graph, sep_set)

        # store resulting data structures
        self.separating_sets_ = sep_set
        self.graph_ = graph

    def test_edge(self, data, X, Y, Z=None):
        if Z is None:
            Z = []
        test_stat, pvalue = self.ci_estimator(data, X, Y, set(Z), **self.ci_estimator_kwargs)
        return test_stat, pvalue

    def learn_skeleton(
        self,
        X: pd.DataFrame,
        graph: Optional[nx.Graph] = None,
        sep_set: Optional[Dict[str, Dict[str, Set[Any]]]] = None,
        fixed_edges: Optional[Set] = None,
    ) -> Tuple[
        nx.Graph,
        Dict[str, Dict[str, Set[Any]]],
        Dict[Any, Dict[Any, float]],
        Dict[Any, Dict[Any, float]],
    ]:
        """Learns the skeleton of a causal DAG using pairwise independence testing.

        Encodes the skeleton via an undirected graph, `networkx.Graph`. Only
        tests with adjacent nodes in the conditioning set.

        Parameters
        ----------
        X : pd.DataFrame
            The data with columns as variables and samples as rows.
        graph : nx.Graph
            The undirected graph containing initialized skeleton of the causal
            relationships.
        sep_set : set
            The separating set.
        fixed_edges : set, optional
            The set of fixed edges. By default, is the empty set.
        return_deps : bool
            Whether to return the two mappings for the dictionary of test statistic
            and pvalues.

        Returns
        -------
        skel_graph : nx.Graph
            The undirected graph of the causal graph's skeleton.
        sep_set : dict of dict of set
            The separating set per pairs of variables.

        Raises
        ------
        ValueError
            If the nodes in the initialization graph do not match the variable
            names in passed in data, ``X``.
        ValueError
            If the nodes in the fixed-edge graph do not match the variable
            names in passed in data, ``X``.

        Notes
        -----
        Learning the skeleton of a causal DAG uses (conditional) independence testing
        to determine which variables are (in)dependent. This specific algorithm
        compares exhaustively pairs of adjacent variables.
        """
        from causal_networkx.discovery.skeleton import learn_skeleton_graph_with_order

        if fixed_edges is None:
            fixed_edges = set()

        # perform pairwise tests to learn skeleton
        skel_graph, sep_set, test_stat_dict, pvalue_dict = learn_skeleton_graph_with_order(  # type: ignore
            X,
            self.ci_estimator,
            adj_graph=graph,
            sep_set=sep_set,
            fixed_edges=fixed_edges,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            keep_sorted=False,
            **self.ci_estimator_kwargs,
        )

        return skel_graph, sep_set, test_stat_dict, pvalue_dict
