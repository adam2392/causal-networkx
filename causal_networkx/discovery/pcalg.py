import logging
from itertools import combinations, permutations
from typing import Any, Callable, Dict, Set, Tuple, Union

import networkx as nx
import pandas as pd

from causal_networkx.cgm import CPDAG, ADMG, PAG
from causal_networkx.discovery.classes import ConstraintDiscovery
from causal_networkx.discovery.skeleton import learn_skeleton_graph_with_order

logger = logging.getLogger()


# TODO: replace PAG with CPDAG object when CPDAG object is made
class MeekRules:
    def _orient_colliders(self, graph: CPDAG, sep_set: Dict[str, Dict[str, Set]]):
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : PAG
            The partial ancestral graph (PAG).
        sep_set : Dict[Dict[Set]]
            The separating set between any two nodes.
        """
        # for every node in the PAG, evaluate neighbors that have any edge
        for u in graph.nodes:
            for v_i, v_j in combinations(graph.adjacencies(u), 2):
                # Check that there is no edge of any type between
                # v_i and v_j, else this is a "shielded" collider.
                # Then check to see if 'u' is in the separating
                # set. If it is not, then there is a collider.
                if not graph.has_adjacency(v_i, v_j) and u not in sep_set[v_i][v_j]:
                    logger.debug(
                        f"orienting collider: {v_i} -> {u} and {v_j} -> {u} to make {v_i} -> {u} <- {v_j}."
                    )

                    if graph.has_undirected_edge(v_i, u):
                        graph.orient_undirected_edge(v_i, u)
                    if graph.has_undirected_edge(v_j, u):
                        graph.orient_undirected_edge(v_j, u)

    def orient_edges(self, skel_graph: CPDAG, sep_set):
        """Orient edges in a skeleton graph to estimate the causal DAG, or CPDAG.

        Uses the separation sets to orient edges via conditional independence
        testing. These are known as the Meek rules.

        Parameters
        ----------
        skel_graph : CPDAG
            A skeleton graph. If ``None``, then will initialize PC using a
            complete graph. By default None.
        sep_set : Dict[Dict[Set]]
            The separating set between any two nodes.
        """
        # dag = skel_graph.to_directed()
        node_ids = skel_graph.nodes()

        # for all pairs of non-adjacent variables with a common neighbor
        # check if we can orient the edge as a collider
        self._orient_colliders(skel_graph, sep_set)
        # check all combinations of nodes and if there is
        # for (i, j) in combinations(node_ids, 2):
        #     adj_i = set(dag.successors(i))
        #     if j in adj_i:
        #         continue
        #     adj_j = set(dag.successors(j))
        #     if i in adj_j:
        #         continue
        #     if sep_set[i][j] is None:
        #         continue
        #     common_k = adj_i & adj_j
        #     for k in common_k:
        #         if k not in sep_set[i][j]:
        #             if dag.has_edge(k, i):
        #                 # _logger.debug('S: remove edge (%s, %s)' % (k, i))
        #                 dag.remove_edge(k, i)
        #             if dag.has_edge(k, j):
        #                 # _logger.debug('S: remove edge (%s, %s)' % (k, j))
        #                 dag.remove_edge(k, j)

        # For all the combination of nodes i and j, apply the following
        # rules.
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:
            change_flag = False
            for (i, j) in permutations(node_ids, 2):
                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                r1_add = self._apply_rule1(skel_graph, i, j)

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                r2_add = self._apply_rule2(skel_graph, i, j)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                r3_add = self._apply_rule3(skel_graph, i, j)

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                # However, this rule is not necessary when the PC-algorithm
                # is used to estimate a DAG.

                if any([r1_add, r2_add, r3_add]) and not change_flag:
                    change_flag = True
            if not change_flag:
                finished = True
                logger.debug(f"Finished applying R1-3, with {idx} iterations")
                break
            idx += 1

        return skel_graph

    def _apply_rule1(self, graph: CPDAG, i, j):
        """Apply rule 1 of Meek's rules.

        Looks for i - j such that k -> i, such that (k,i,j)
        is an unshielded triple. Then can orient i - j as i -> j.
        """
        added_arrows = False

        # Check if i-j.
        if graph.has_undirected_edge(i, j):
            for k in graph.predecessors(i):
                # for k in graph.adjacencies(i):
                #     # Skip if there is an arrow i->k.
                #     if graph.has_edge(i, k):
                #         continue
                #     # Skip if k and j are adjacent because then it is a
                #     # shielded triple
                if graph.has_adjacency(k, j):
                    continue

                # Make i-j into i->j
                logger.debug(f"R1: Removing edge ({j}, {j}) and orienting as {i} -> {j}.")
                graph.orient_undirected_edge(i, j)

                added_arrows = True
                break
        return added_arrows

    def _apply_rule2(self, graph: CPDAG, i, j):
        """Apply rule 2 of Meek's rules.

        Check for i - j, and then looks for i -> k -> j
        triple, to orient i - j as i -> j.
        """
        added_arrows = False

        # Check if i-j.
        if graph.has_undirected_edge(i, j):
            # Find nodes k where k is i->k
            succs_i = set()
            for k in graph.successors(i):
                if not graph.has_edge(k, i):
                    succs_i.add(k)
            # Find nodes j where j is k->j.
            preds_j = set()
            for k in graph.predecessors(j):
                if not graph.has_edge(j, k):
                    preds_j.add(k)
            # Check if there is any node k where i->k->j.
            if len(succs_i.intersection(preds_j)) > 0:
                # Make i-j into i->j
                logger.debug(f"R2: Removing edge {i}-{j} to form {i}->{j}.")
                graph.orient_undirected_edge(i, j)
                added_arrows = True
        return added_arrows

    def _apply_rule3(self, graph: CPDAG, i, j):
        """Apply rule 3 of Meek's rules.

        Check for i - j, and then looks for k -> j <- l
        collider, and i - k and i - l, then orient i -> j.
        """
        added_arrows = False

        # Check if i-j first
        if graph.has_undirected_edge(i, j):
            # For all the pairs of nodes adjacent to i,
            # look for (k, l), such that j -> l and k -> l
            for (k, l) in combinations(graph.adjacencies(i), 2):
                # Skip if k and l are adjacent.
                if graph.has_adjacency(k, l):
                    continue
                # Skip if not k->j.
                if graph.has_edge(j, k) or (not graph.has_edge(k, j)):
                    continue
                # Skip if not l->j.
                if graph.has_edge(j, l) or (not graph.has_edge(l, j)):
                    continue

                # if i - k and i - l, then  at this point, we have a valid path
                # to orient
                if graph.has_undirected_edge(k, i) and graph.has_undirected_edge(l, i):
                    logger.debug(f"R3: Removing edge {i}-{j} to form {i}->{j}")
                    graph.orient_undirected_edge(i, j)
                    added_arrows = True
                    break
        return added_arrows


class PC(ConstraintDiscovery, MeekRules):
    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        apply_orientations: bool = True,
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
        min_cond_set_size : int, optional
            Minimum size of the conditioning set, by default None, which will be set to '0'.
            Used to constrain the computation spent on the algorithm.
        max_cond_set_size : int, optional
            Maximum size of the conditioning set, by default None. Used to limit
            the computation spent on the algorithm.
        max_iter : int
            The maximum number of iterations through the graph to apply
            orientation rules.
        apply_orientations : bool
            Whether or not to apply orientation rules given the learned skeleton graph
            and separating set per pair of variables.
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
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size=min_cond_set_size,
            max_cond_set_size=max_cond_set_size,
            **ci_estimator_kwargs,
        )
        self.max_iter = max_iter
        self.apply_orientations = apply_orientations

    def fit(self, X: pd.DataFrame) -> None:
        """Fit PC algorithm on dataset 'X'."""
        # initialize graph
        graph, sep_set, fixed_edges = self._initialize_graph(X)

        # learn skeleton graph and the separating sets per variable
        graph, sep_set = self.learn_skeleton(X, graph, sep_set, fixed_edges)

        self.separating_sets_ = sep_set
        # convert Graph object to a CPDAG object with
        # all undirected edges
        graph = CPDAG(incoming_uncertain_data=graph)

        # orient edges into a CPDAG
        if self.apply_orientations:
            graph = self.orient_edges(graph, sep_set)

        self.graph_ = graph


class PC1(PC):
    """PC-(1) algorithm variant.

    This constrains the PC algorithm to only run (X, Y) given Z
    conditional independence once for every iteration through the
    cardinality of Z.
    """

    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, ADMG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = 1,
        apply_orientations: bool = False,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size=min_cond_set_size,
            max_cond_set_size=max_cond_set_size,
            max_iter=max_iter,
            max_combinations=max_combinations,
            apply_orientations=apply_orientations,
            **ci_estimator_kwargs,
        )

    def learn_skeleton(
        self,
        X: pd.DataFrame,
        graph: nx.Graph,
        sep_set: Dict[str, Dict[str, Set[Any]]],
        fixed_edges: Set = set(),
    ) -> Tuple[nx.Graph, Dict[str, Dict[str, Set]], Dict[str, Dict[str, float]]]:
        # perform pairwise tests to learn skeleton
        skel_graph, sep_set, parent_dep_dict = learn_skeleton_graph_with_order(
            X,
            graph,
            sep_set,
            self.ci_estimator,
            fixed_edges,
            self.alpha,
            max_cond_set_size=self.max_cond_set_size,
            min_cond_set_size=self.min_cond_set_size,
            max_combinations=self.max_combinations,
            **self.ci_estimator_kwargs,
        )
        self.parent_dep_dict_ = parent_dep_dict
        return skel_graph, sep_set, parent_dep_dict
