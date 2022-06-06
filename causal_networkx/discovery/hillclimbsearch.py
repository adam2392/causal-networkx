from typing import Any, Callable, Optional, Union

import networkx as nx
import pandas as pd
from tqdm.auto import trange

from causal_networkx import ADMG


class ScoreBasedDiscovery:
    graph_: Optional[Any]

    def __init__(
        self,
        scoring_method: Union[Callable, str] = "bic",
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        tabu_length: int = 100,
        max_indegree: int = None,
        epsilon: float = 1e-4,
        max_iter: int = 1e6,
        **scoring_method_kwargs,
    ) -> None:

        self.epsilon = epsilon
        self.max_iter = max_iter
        self.scoring_method = scoring_method
        self.scoring_method_kwargs = scoring_method_kwargs
        self.init_graph = init_graph
        self.fixed_edges = fixed_edges

        self.graph_ = None
        self.tabu_length = tabu_length
        self.max_indegree = max_indegree


def bic_score(X: pd.DataFrame, node, parents):
    if node not in X.columns:
        raise RuntimeError(f"Node {node} not in the dataset of X.")
    if any(parent not in X.columns for parent in parents):
        raise RuntimeError(f"Some of {parents} are not in dataset X.")

    # node_count = len(X.columns)

    # counts = np.array()

    # compute log likelihoods
    # log_likelihoods -= log_conditionals
    # log_likelihoods *= counts

    # # compute the final BIC score
    # penalty_factor = 0.5 * np.log(sample_size) * num_parents_states * (node_count - 1)
    # score = np.sum(log_likelihoods) - penalty_factor

    # return score


class HillClimbSearch(ScoreBasedDiscovery):
    def __init__(
        self,
        scoring_method: Callable,
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        tabu_length: int = 100,
        max_indegree: int = None,
        epsilon: float = 0.0001,
        max_iter: int = 1000000,
        **scoring_method_kwargs,
    ) -> None:
        super().__init__(
            scoring_method,
            init_graph,
            fixed_edges,
            tabu_length,
            max_indegree,
            epsilon,
            max_iter,
            **scoring_method_kwargs,
        )

    def legal_operations(self):
        pass

    def fit(self, X: pd.DataFrame):
        # score_func = self.scoring_method

        nodes = X.columns

        # stores previously visited solutions to prevent recomputing
        tabu_list = []

        # initialize the starting graph, which is empty
        # as opposed to complete in the constraint-based case.
        graph = self.init_graph
        if graph is None:
            graph = nx.empty_graph(nodes)
        else:
            if any(node not in graph for node in nodes):
                raise RuntimeError(
                    "The initial graph is missing some nodes present in the dataset."
                )

        # initialize possibly fixed edges
        if self.fixed_edges is not None:
            for edge in self.fixed_edges.edges:
                graph.add_edge(edge)

        if not nx.is_directed_acyclic_graph(graph):
            raise RuntimeError("Fixed edges creates a cycle, which is not allowed.")

        # Now iterate to find the best scoring operation
        # for the current model.

        for idx in trange(self.max_iter):
            # get the best operation and the delta of that score
            best_operation, best_score_delta = 0, 0

            # check if we reached end condition
            if best_operation is None or best_score_delta < self.epsilon:
                break
            elif best_operation[0] == "+":
                # add an edge
                graph.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                # remove an edge
                graph.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                # flip an edge
                X, Y = best_operation[1]
                graph.remove_edge(X, Y)
                graph.add_edge(Y, X)
                tabu_list.append(best_operation)

        return graph
