import itertools
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Set, Union

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx import ADMG, PAG
from causal_networkx.discovery.skeleton import learn_skeleton_graph


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
    init_graph : nx.Graph | CausalGraph, optional
        An initialized graph. If ``None``, then will initialize PC using a
        complete graph. By default None.
    fixed_edges : nx.Graph, optional
        An undirected graph with fixed edges. If ``None``, then will initialize PC using a
        complete graph. By default None.
    max_cond_set_size : int, optional
        Maximum size of the conditioning set, by default None. Used to limit
        the computation spent on the algorithm.
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

    graph_: Optional[PAG]
    separating_sets_: Optional[Dict[str, Dict[str, Set[Any]]]]

    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        max_cond_set_size: int = None,
        **ci_estimator_kwargs,
    ):
        self.alpha = alpha
        self.ci_estimator = ci_estimator
        self.ci_estimator_kwargs = ci_estimator_kwargs
        self.init_graph = init_graph
        self.fixed_edges = fixed_edges

        if max_cond_set_size is None:
            max_cond_set_size = np.inf
        self.max_cond_set_size = max_cond_set_size

        self.separating_sets_ = None
        self.graph_ = None

    def _initialize_graph(self, X):
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

    def _learn_skeleton_from_neighbors(
        self,
        X: pd.DataFrame,
        graph: nx.Graph,
        sep_set: Dict[str, Dict[str, Set[Any]]],
        fixed_edges: Set = set(),
    ):
        """Learns the skeleton of a causal DAG using pairwise independence testing.

        Encodes the skeleton via an undirected graph, `nx.Graph`. Only
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
        """
        # perform pairwise tests to learn skeleton
        skel_graph, sep_set = learn_skeleton_graph(
            X,
            graph,
            sep_set,
            self.ci_estimator,
            fixed_edges,
            self.alpha,
            max_cond_set_size=self.max_cond_set_size,
            **self.ci_estimator_kwargs,
        )
        return skel_graph, sep_set
