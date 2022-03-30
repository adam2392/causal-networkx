from typing import Callable
from itertools import combinations, permutations

import numpy as np
import networkx as nx
import pandas as pd


class PC:
    def __init__(
        self,
        estimator: Callable,
        alpha: float = 0.05,
        init_graph: nx.Graph = None,
        fixed_edges: nx.Graph = None,
        max_cond_set_size: int = None,
        **estimator_kwargs,
    ):
        self.alpha = alpha
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs
        self.init_graph = init_graph
        self.fixed_edges = fixed_edges

        if max_cond_set_size is None:
            max_cond_set_size = np.inf
        self.max_cond_set_size = max_cond_set_size

    def fit(self, X: pd.DataFrame) -> None:
        nodes = X.columns

        # keep track of separating sets
        sep_set = [[set() for _ in range(len(nodes))] for _ in range(len(nodes))]

        # initialize the starting graph
        if self.init_graph is None:
            graph = nx.complete_graph(nodes)
        else:
            graph = self.init_graph

            if graph.nodes != nodes:
                raise ValueError(
                    f"The nodes within the initial graph, {graph.nodes}, "
                    f"do not match the nodes in the passed in data, {nodes}."
                )

            # since we are not starting from a complete graph,
            # find the separating sets
            for (node_i, node_j) in combinations(graph.nodes):
                if not graph.has_edge(node_i, node_j):
                    sep_set[node_i][node_j] = None
                    sep_set[node_j][node_i] = None

        # check on fixed edges and keep track
        fixed_edges = set()
        if self.fixed_edges is not None:
            if self.fixed_edges.nodes != nodes:
                raise ValueError(
                    f"The nodes within the fixed-edges graph, {self.fixed_edges.nodes}, "
                    f"do not match the nodes in the passed in data, {nodes}."
                )

            for (i, j) in self.fixed_edges.edges:
                fixed_edges.add((i, j))
                fixed_edges.add((j, i))

        # perform pairwise tests
        graph = self._learn_graph(X, graph, fixed_edges)
        self.graph_ = graph

    def _learn_graph(self, X, graph, fixed_edges):
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

        Returns
        -------
        graph : networkx.Graph
            The discovered graph from data.
        """
        nodes = graph.nodes
        sep_set = self.sep_set

        size_cond_set = 0
        while 1:
            remove_edges = []

            # loop through all possible permutation of
            # two nodes in the graph
            for (i, j) in permutations(nodes, 2):
                if (i, j) in fixed_edges:
                    continue

                # get all neighbors for "i" that are
                # also adjacent to "j"
                adj_i = list(graph.neighbors(i))
                if j not in adj_i:
                    continue
                adj_i.remove(j)

                if len(adj_i) >= size_cond_set:
                    if len(adj_i) < size_cond_set:
                        continue

                    # loop through all conditioning sets
                    for cond_set in combinations(adj_i, size_cond_set):
                        # compute conditional independence test
                        pvalue = self.estimator(
                            X, i, j, set(cond_set), **self.estimator_kwargs
                        )

                        # two variables found to be independent
                        if pvalue > self.alpha:
                            if graph.has_edge(i, j):
                                remove_edges.appendn((i, j))

                            sep_set[i][j] |= set(cond_set)
                            sep_set[j][i] |= set(cond_set)

                    size_cond_set += 1

                    if size_cond_set > self.max_cond_set_size:
                        break
                else:
                    break

            # finally remove edges after performing
            # conditional independence tests
            graph.remove_edges_from(remove_edges)
        return graph
