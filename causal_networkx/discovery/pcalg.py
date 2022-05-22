import logging
from itertools import combinations, permutations
from typing import Any, Callable, Dict, Set, Tuple, Union

import networkx as nx
import pandas as pd

from causal_networkx import CPDAG, DAG

from .classes import ConstraintDiscovery

logger = logging.getLogger()


class PC(ConstraintDiscovery):
    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, DAG, CPDAG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = None,
        apply_orientations: bool = True,
        **ci_estimator_kwargs,
    ):
        """Peter and Clarke (PC) algorithm for causal discovery.

        Assumes causal sufficiency, that is, all confounders in the
        causal graph are observed variables. See :footcite:`Spirtes1993` for
        full details on the algorithm.

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
        max_combinations : int, optional
            Maximum number of tries with a conditioning set chosen from the set of possible
            parents still, by default None. If None, then will not be used. If set, then
            the conditioning set will be chosen lexographically based on the sorted
            test statistic values of 'ith Pa(X) -> X', for each possible parent node of 'X'.
        apply_orientations : bool
            Whether or not to apply orientation rules given the learned skeleton graph
            and separating set per pair of variables. If ``True`` (default), will
            apply Meek's orientation rules R0-3, orienting colliders and certain
            arrowheads :footcite:`Meek1995`.
        ci_estimator_kwargs : dict
            Keyword arguments for the ``ci_estimator`` function.

        Attributes
        ----------
        graph_ : CPDAG
            The graph discovered.
        separating_sets_ : dict
            The dictionary of separating sets, where it is a nested dictionary from
            the variable name to the variable it is being compared to the set of
            variables in the graph that separate the two.

        References
        ----------
        .. footbibliography::
        """
        super().__init__(
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size=min_cond_set_size,
            max_cond_set_size=max_cond_set_size,
            max_combinations=max_combinations,
            **ci_estimator_kwargs,
        )
        self.max_iter = max_iter
        self.apply_orientations = apply_orientations

    def learn_skeleton(
        self,
        X: pd.DataFrame,
        graph: nx.Graph = None,
        sep_set: Dict[str, Dict[str, Set[Any]]] = None,
        fixed_edges: Set = set(),
    ) -> Tuple[nx.Graph, Dict[str, Dict[str, Set[Any]]]]:
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

        Notes
        -----
        Learning the skeleton of a causal DAG uses (conditional) independence testing
        to determine which variables are (in)dependent. This specific algorithm
        compares exhaustively pairs of adjacent variables.
        """
        from causal_networkx.discovery.skeleton import (
            learn_skeleton_graph_with_neighbors,
        )

        # perform pairwise tests to learn skeleton
        skel_graph, sep_set = learn_skeleton_graph_with_neighbors(
            X,
            self.ci_estimator,
            adj_graph=graph,
            sep_set=sep_set,
            fixed_edges=fixed_edges,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            **self.ci_estimator_kwargs,
        )
        return skel_graph, sep_set

    def convert_skeleton_graph(self, graph: nx.Graph) -> CPDAG:
        """Convert skeleton graph as undirected networkx Graph to CPDAG.

        Parameters
        ----------
        graph : nx.Graph
            Converts a skeleton graph to the representation needed
            for PC algorithm, a CPDAG.

        Returns
        -------
        graph : CPDAG
            The CPDAG class.
        """
        # convert Graph object to a CPDAG object with
        # all undirected edges
        graph = CPDAG(incoming_uncertain_data=graph)
        return graph

    def orient_edges(self, skel_graph: CPDAG, sep_set) -> CPDAG:
        """Orient edges in a skeleton graph to estimate the causal DAG, or CPDAG.

        Uses the separation sets to orient edges via conditional independence
        testing. These are known as the Meek rules :footcite:`Meek1995`.

        Parameters
        ----------
        skel_graph : causal_networkx.CPDAG
            A skeleton graph. If ``None``, then will initialize PC using a
            complete graph. By default None.
        sep_set : Dict[Dict[Set]]
            The separating set between any two nodes.
        """
        node_ids = skel_graph.nodes()

        # for all pairs of non-adjacent variables with a common neighbor
        # check if we can orient the edge as a collider
        self._orient_colliders(skel_graph, sep_set)

        # For all the combination of nodes i and j, apply the following
        # rules.
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:  # type: ignore
            change_flag = False
            for (i, j) in permutations(node_ids, 2):
                if i == j:
                    continue
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

    def _orient_colliders(self, graph: CPDAG, sep_set: Dict[str, Dict[str, Set]]):
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : CPDAG
            The CPDAG.
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

    def _apply_rule1(self, graph: CPDAG, i, j):
        """Apply rule 1 of Meek's rules.

        Looks for i - j such that k -> i, such that (k,i,j)
        is an unshielded triple. Then can orient i - j as i -> j.
        """
        added_arrows = False

        # Check if i-j.
        if graph.has_undirected_edge(i, j):
            for k in graph.predecessors(i):
                # Skip if k and j are adjacent because then it is a
                # shielded triple
                if graph.has_adjacency(k, j):
                    continue

                # Make i-j into i->j
                logger.debug(f"R1: Removing edge ({i}, {j}) and orienting as {k} -> {i} -> {j}.")
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
