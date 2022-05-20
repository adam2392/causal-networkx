from itertools import combinations
from typing import Callable, Dict, Set, Tuple, Union

import networkx as nx
import pandas as pd

from causal_networkx import ADMG
from causal_networkx.discovery.classes import ConstraintDiscovery


def _has_both_edges(dag, i, j):
    return dag.has_edge(i, j) and dag.has_edge(j, i)


def _has_any_edge(dag, i, j):
    return dag.has_edge(i, j) or dag.has_edge(j, i)


class PC(ConstraintDiscovery):
    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        max_cond_set_size: int = None,
        **ci_estimator_kwargs,
    ):
        """Peter and Clarke (PC) algorithm for causal discovery.

        Assumes causal sufficiency, that is, all confounders in the
        causal graph are observed variables.

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
        super().__init__(
            ci_estimator, alpha, init_graph, fixed_edges, max_cond_set_size, **ci_estimator_kwargs
        )

    def learn_skeleton(self, X: pd.DataFrame) -> Tuple[nx.Graph, Dict[str, Dict[str, Set]]]:
        """Learn skeleton from data.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset.

        Returns
        -------
        skel_graph : nx.Graph
            The skeleton graph.
        sep_set : Dict[str, Dict[str, Set]]
            The separating set.
        """
        graph, sep_set, fixed_edges = self._initialize_graph(X)
        skel_graph, sep_set = self._learn_skeleton_from_neighbors(X, graph, sep_set, fixed_edges)
        return skel_graph, sep_set

    def fit(self, X: pd.DataFrame) -> None:
        """Fit PC algorithm on dataset 'X'."""
        # learn skeleton
        skel_graph, sep_set = self.learn_skeleton(X)

        # perform CI tests to orient edges into a DAG
        graph = self._orient_edges(skel_graph, sep_set)

        self.separating_sets_ = sep_set
        self.graph_ = graph

    def _orient_edges(self, skel_graph, sep_set):
        """Orient edges in a skeleton graph to estimate the causal DAG, or CPDAG.

        Uses the separation sets to orient edges via conditional independence
        testing.

        Parameters
        ----------
        skel_graph : nx.Graph
            A skeleton graph. If ``None``, then will initialize PC using a
            complete graph. By default None.
        sep_set : _type_
            _description_
        """
        dag = skel_graph.to_directed()
        node_ids = skel_graph.nodes()
        for (i, j) in combinations(node_ids, 2):
            adj_i = set(dag.successors(i))
            if j in adj_i:
                continue
            adj_j = set(dag.successors(j))
            if i in adj_j:
                continue
            if sep_set[i][j] is None:
                continue
            common_k = adj_i & adj_j
            for k in common_k:
                if k not in sep_set[i][j]:
                    if dag.has_edge(k, i):
                        # _logger.debug('S: remove edge (%s, %s)' % (k, i))
                        dag.remove_edge(k, i)
                    if dag.has_edge(k, j):
                        # _logger.debug('S: remove edge (%s, %s)' % (k, j))
                        dag.remove_edge(k, j)
        # For all the combination of nodes i and j, apply the following
        # rules.
        old_dag = dag.copy()
        while True:
            for (i, j) in combinations(node_ids, 2):
                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Look all the predecessors of i.
                    for k in dag.predecessors(i):
                        # Skip if there is an arrow i->k.
                        if dag.has_edge(i, k):
                            continue
                        # Skip if k and j are adjacent.
                        if _has_any_edge(dag, k, j):
                            continue
                        # Make i-j into i->j
                        # _logger.debug("R1: remove edge (%s, %s)" % (j, i))
                        dag.remove_edge(j, i)
                        break

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Find nodes k where k is i->k.
                    succs_i = set()
                    for k in dag.successors(i):
                        if not dag.has_edge(k, i):
                            succs_i.add(k)
                    # Find nodes j where j is k->j.
                    preds_j = set()
                    for k in dag.predecessors(j):
                        if not dag.has_edge(j, k):
                            preds_j.add(k)
                    # Check if there is any node k where i->k->j.
                    if len(succs_i & preds_j) > 0:
                        # Make i-j into i->j
                        # _logger.debug("R2: remove edge (%s, %s)" % (j, i))
                        dag.remove_edge(j, i)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                #
                # Check if i-j.
                if _has_both_edges(dag, i, j):
                    # Find nodes k where i-k.
                    adj_i = set()
                    for k in dag.successors(i):
                        if dag.has_edge(k, i):
                            adj_i.add(k)
                    # For all the pairs of nodes in adj_i,
                    for (k, l) in combinations(adj_i, 2):
                        # Skip if k and l are adjacent.
                        if _has_any_edge(dag, k, l):
                            continue
                        # Skip if not k->j.
                        if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                            continue
                        # Skip if not l->j.
                        if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                            continue
                        # Make i-j into i->j.
                        # _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                        dag.remove_edge(j, i)
                        break

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                # However, this rule is not necessary when the PC-algorithm
                # is used to estimate a DAG.

            if nx.is_isomorphic(dag, old_dag):
                break
            old_dag = dag.copy()

        return dag
