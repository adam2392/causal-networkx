import logging
from itertools import combinations, permutations
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx import PAG, CausalGraph
from causal_networkx.algorithms.pag import discriminating_path, uncovered_pd_path
from causal_networkx.discovery.classes import ConstraintDiscovery

logger = logging.getLogger()


class FCI(ConstraintDiscovery):
    def __init__(
        self,
        ci_estimator: Callable,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, CausalGraph] = None,
        fixed_edges: nx.Graph = None,
        max_cond_set_size: int = None,
        max_path_length: int = None,
        selection_bias: bool = False,
        augmented: bool = False,
        max_iter: int = 1000,
        **ci_estimator_kwargs,
    ):
        """The Fast Causal Inference (FCI) algorithm for causal discovery.

        A complete constraint-based causal discovery algorithm that
        operates on observational data :footcite:`Zhang2008`.

        Parameters
        ----------
        ci_estimator : Callable
            _description_
        alpha : float, optional
            _description_, by default 0.05
        init_graph : Union[nx.Graph, CausalGraph], optional
            _description_, by default None
        fixed_edges : nx.Graph, optional
            _description_, by default None
        max_cond_set_size : int, optional
            _description_, by default None
        max_path_length : int, optional
            The maximum length of any discriminating path, or None if unlimited.
        selection_bias : bool
            Whether or not to account for selection bias within the causal PAG.
            See [1].
        augmented : bool
            Whether or not to run the augmented version of FCI. See [1].

        References
        ----------
        .. footbibliography::

        Notes
        -----
        Note that the algorithm is called "fast causal inference", but in reality
        the algorithm is quite expensive in terms of the number of conditional
        independence tests it must run.
        """
        super().__init__(
            ci_estimator, alpha, init_graph, fixed_edges, max_cond_set_size, **ci_estimator_kwargs
        )

        if max_path_length is None:
            max_path_length = np.inf
        self.max_path_length = max_path_length
        self.selection_bias = selection_bias
        self.augmented = augmented
        self.max_iter = max_iter

    def _orient_colliders(self, graph: PAG, sep_set: Dict[str, Dict[str, Set]]):
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
            for v_i, v_j in combinations(graph.neighbors(u), 2):
                # Check that there is no edge of any type between
                # v_i and v_j, else this is a "shielded" collider.
                # Then check to see if 'u' is in the separating
                # set. If it is not, then there is a collider.
                if not graph.has_adjacency(v_i, v_j) and u not in sep_set[v_i][v_j]:
                    logger.debug(
                        f"orienting collider: {v_i} -> {u} and {v_j} -> {u} to make {v_i} -> {u} <- {v_j}."
                    )

                    if graph.has_circle_edge(v_i, u):
                        graph.orient_circle_edge(v_i, u, "arrow")
                    if graph.has_circle_edge(v_j, u):
                        graph.orient_circle_edge(v_j, u, "arrow")
                # else:
                # definite non-collider
                # test = 1

    def _apply_rule1(self, graph: PAG, u, a, c) -> bool:
        """Apply rule 1 of the FCI algorithm.

        If A *-> u o-* C, A and C are not adjacent,
        then we can orient the triple as A *-> u -> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False

        # If A *-> u o-* C, A and C are not adjacent,
        # then we can orient the triple as A *-> u -> C.
        # check that a and c are not adjacent
        if not graph.has_adjacency(a, c):
            # check a *-> u o-* c
            if (graph.has_edge(a, u) or graph.has_bidirected_edge(a, u)) and graph.has_circle_edge(
                c, u
            ):
                logger.debug(f"Rule 1: Orienting edge {u} o-* {c} to {u} -> {c}.")
                # orient the edge from u to c and delete
                # the edge from c to u
                if graph.has_circle_edge(u, c):
                    graph.orient_circle_edge(u, c, "arrow")
                graph.orient_circle_edge(c, u, "tail")
                added_arrows = True

        return added_arrows

    def _apply_rule2(self, graph: PAG, u, a, c) -> bool:
        """Apply rule 2 of FCI algorithm.

        If

        - A -> u *-> C, or A *-> u -> C, and
        - A *-o C,

        then orient A *-> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False
        # check that a *-o c edge exists
        if graph.has_circle_edge(a, c):
            # - A -> u *-> C, or A *-> u -> C, and
            # - A *-o C,
            # check for A -> u and check that u *-> c
            condition_one = (
                graph.has_edge(a, u)
                and not graph.has_edge(u, a)
                and not graph.has_circle_edge(u, a)
                and graph.edge_type(u, c) in ["arrow", "bidirected"]
            )

            # check that a *-> u -> c
            condition_two = (
                graph.edge_type(a, u) in ["arrow", "bidirected"]
                and graph.edge_type(u, c) == "arrow"
                and not graph.has_edge(c, u)
                and not graph.has_circle_edge(c, u)
            )

            if condition_one or condition_two:
                logger.debug(f"Rule 2: Orienting circle edge to {a} -> {c}")
                # orient a *-> c
                graph.orient_circle_edge(a, c, "arrow")
                added_arrows = True
        return added_arrows

    def _apply_rule3(self, graph: PAG, u, a, c) -> bool:
        """Apply rule 3 of FCI algorithm.

        If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
        and v *-o u, then orient v *-> u.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False
        # check that a and c are not adjacent
        if not graph.has_adjacency(a, c):
            # If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
            # and v *-o u, then orient v *-> u.
            # check that a *-> u <-* c
            condition_one = (graph.has_edge(a, u) or graph.has_bidirected_edge(a, u)) and (
                graph.has_edge(c, u) or graph.has_bidirected_edge(c, u)
            )
            if not condition_one:  # add quick check here to skip non-relevant u nodes
                return added_arrows

            # check for all other neighbors to find a 'v' node
            # with the structure A *-o v o-* C
            for v in graph.neighbors(u):
                # check that v is not a, or c
                if v in (a, c):
                    continue

                # check that v *-o u
                if not graph.has_circle_edge(v, u):
                    continue

                # check that a *-o v o-* c
                condition_two = graph.has_circle_edge(a, v) and graph.has_circle_edge(c, v)
                if condition_one and condition_two:
                    logger.debug(f"Rule 3: Orienting {v} -> {u}.")
                    graph.orient_circle_edge(v, u, "arrow")
                    added_arrows = True
        return added_arrows

    def _apply_rule4(self, graph: PAG, u, a, c, sep_set) -> Tuple[bool, Dict]:
        """Apply rule 4 of FCI algorithm.

        If a path, U = <v, ..., a, u, c> is a discriminating
        path between v and c for u, u o-* c, u in SepSet(v, c),
        orient u -> c. Else, orient a <-> u <-> c.

        A discriminating path, p, is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        Parameters
        ----------
        graph : PAG
            PAG to orient.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.
        sep_set : set
            The separating set to check.

        Notes
        -----
        ...
        """
        added_arrows = False
        explored_nodes: Dict[Any, None] = dict()

        # a must point to c for us to begin a discriminating path and
        # not be bi-directional
        if not graph.has_edge(a, c) or graph.has_bidirected_edge(c, a):
            return added_arrows, explored_nodes

        # c must also point to u with a circle edge
        # check u o-* c
        if not graph.has_circle_edge(c, u):
            return added_arrows, explored_nodes

        # 'a' cannot be a definite collider if there is no arrow pointing from
        # u to a either as: u -> a, or u <-> a
        if not graph.has_edge(u, a) and not graph.has_bidirected_edge(u, a):
            return added_arrows, explored_nodes

        explored_nodes, found_discriminating_path, disc_path = discriminating_path(
            graph, u, a, c, self.max_path_length
        )
        disc_path_str = " ".join(
            [
                graph.print_edge(disc_path[idx], disc_path[idx + 1])
                for idx in range(len(disc_path) - 1)
            ]
        )
        if found_discriminating_path:
            last_node = list(explored_nodes.keys())[-1]

            # now check if u is in SepSet(v, c)
            # handle edge case where sep_set is empty.
            if last_node in sep_set:
                if u in sep_set[last_node][c]:
                    # orient u -> c
                    graph.remove_circle_edge(c, u)
                if graph.has_circle_edge(u, c):
                    graph.orient_circle_edge(u, c, "arrow")
                logger.debug(f"Rule 4: orienting {u} -> {c}.")
                logger.debug(disc_path_str)
            else:
                # orient u <-> c
                if graph.has_circle_edge(u, c):
                    graph.orient_circle_edge(u, c, "arrow")
                if graph.has_circle_edge(c, u):
                    graph.orient_circle_edge(c, u, "arrow")
                logger.debug(f"Rule 4: orienting {u} <-> {c}.")
                logger.debug(disc_path_str)
            added_arrows = True

        return added_arrows, explored_nodes

    def _apply_rule8(self, graph: PAG, u, a, c) -> bool:
        """Apply rule 8 of FCI algorithm.

        If A -> u -> C, or A -o B -> C
        and A o-> C, then orient A o-> C as A -> C.

        Without dealing with selection bias, we ignore
        A -o B -> C condition.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        # If A -> u -> C,
        # and A o-> C, then orient A o-> C as A -> C.
        added_arrows = False

        # First check that A o-> C
        if graph.has_circle_edge(c, a) and graph.has_edge(a, c):
            # check that A -> u -> C
            condition_one = graph.has_edge(a, u) and not graph.has_circle_edge(u, a)
            condition_two = graph.has_edge(u, c) and not graph.has_circle_edge(c, u)

            if condition_one and condition_two:
                logger.debug(f"Rule 8: Orienting {a} o-> {c} as {a} -> {c}.")
                # now orient A o-> C as A -> C
                graph.orient_circle_edge(c, a, "tail")
                added_arrows = True
        return added_arrows

    def _apply_rule9(self, graph: PAG, u, a, c) -> Tuple[bool, List]:
        """Apply rule 9 of FCI algorithm.

        If A o-> C and p = <A, u, v, ..., C> is an uncovered
        possibly directed path from A to C such that u and C
        are not adjacent, orient A o-> C  as A -> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        uncov_path : list
            The uncovered potentially directed path from 'a' to 'c' through 'u'.
        """
        added_arrows = False
        uncov_path: List[Any] = []

        # Check A o-> C and # check that u is not adjacent to c
        if (graph.has_circle_edge(c, a) and graph.has_edge(a, c)) and not graph.has_adjacency(u, c):
            # check that <a, u> is potentially directed
            if graph.has_edge(a, u):
                # check that A - u - v, ..., c is an uncovered pd path
                uncov_path, path_exists = uncovered_pd_path(
                    graph, u, c, max_path_length=self.max_path_length, first_node=a
                )

                # orient A o-> C to A -> C
                if path_exists:
                    logger.debug(f"Rule 9: Orienting edge {a} o-> {c} to {a} -> {c}.")
                    graph.orient_circle_edge(c, a, "tail")
                    added_arrows = True

        return added_arrows, uncov_path

    def _apply_rule10(self, graph: PAG, u, a, c) -> Tuple[bool, List]:
        """Apply rule 10 of FCI algorithm.

        If A o-> C and u -> C <- v and

        - p1 is an uncovered pd path from A to u
        - p2 is an uncovered pd path from A to v

        Then say m is adjacent to A on p1 (could be u).
        Say w is adjacent to A on p2 (could be v).

        If m and w are distinct and not adjacent, then
        orient A o-> C  as A -> C.

        Parameters
        ----------
        graph : PAG
            The causal graph to apply rules to.
        u : node
            A node in the graph.
        a : node
            A node in the graph.
        c : node
            A node in the graph.

        Returns
        -------
        added_arrows : bool
            Whether or not arrows were modified in the graph.
        """
        added_arrows = False
        uncov_pd_path = []

        # Check A o-> C
        if graph.has_circle_edge(c, a) and graph.has_edge(a, c):
            # check that u -> C
            if graph.has_edge(u, c) and not graph.has_circle_edge(c, u):
                # loop through all adjacent neighbors of c now to get
                # possible 'v' node
                for idx, v in enumerate(graph.neighbors(c)):
                    if v in (a, u):
                        continue

                    # make sure v -> C and not v o-> C
                    if not graph.has_edge(v, c) or graph.has_circle_edge(c, v):
                        continue

                    # At this point, we want the paths from A to u and A to v
                    # to begin with a distinct m and w node, else we will not
                    # apply R10. Thus, we will get all 2-pairs of neighbors of A
                    # that:
                    # i) begin the uncovered pd path and
                    # ii) are distinct (done by construction) here
                    for (m, w) in combinations(graph.neighbors(a)):  # type: ignore
                        if m == c or w == c:
                            continue

                        # m and w must be on a potentially directed path
                        if not graph.has_edge(a, m) or not graph.has_edge(a, w):
                            continue

                        # we do not know which path a-u or a-v, m and w are on
                        # so we must traverse the graph in both directions
                        # get the uncovered pd path from A to u just once
                        found_uncovered_a_to_v = False
                        a_to_u_path, found_uncovered_a_to_u = uncovered_pd_path(
                            graph, a, u, max_path_length=self.max_path_length, second_node=m
                        )
                        if not found_uncovered_a_to_u:
                            a_to_u_path, found_uncovered_a_to_u = uncovered_pd_path(
                                graph, a, u, max_path_length=self.max_path_length, second_node=w
                            )
                            # if we don't have an uncovered pd path here, then no point in looking
                            # for other paths
                            if found_uncovered_a_to_u:
                                a_to_v_path, found_uncovered_a_to_v = uncovered_pd_path(
                                    graph, a, v, max_path_length=self.max_path_length, second_node=m
                                )
                        else:
                            a_to_v_path, found_uncovered_a_to_v = uncovered_pd_path(
                                graph, a, v, max_path_length=self.max_path_length, second_node=w
                            )

                        # if we have not found another path, then just continue
                        if not found_uncovered_a_to_v:
                            continue

                        # at this point, we have an uncovered path from a to u and a to v
                        # with a distinct second node on both paths
                        # orient A o-> C to A -> C
                        logger.debug(f"Rule 10: Orienting edge {a} o-> {c} to {a} -> {c}.")
                        graph.orient_circle_edge(c, a, "tail")
                        added_arrows = True
                        uncov_pd_path = a_to_u_path

        return added_arrows, uncov_pd_path

    def _apply_rules_1to10(self, graph: PAG, sep_set: Dict[str, Dict[str, Set[Any]]]):
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:
            change_flag = False
            logger.debug(f"Running R1-10 for iteration {idx}")

            for u in graph.nodes:
                for (a, c) in permutations(graph.neighbors(u), 2):
                    # if u == 'x4':
                    #     print(u, a, c)
                    # apply R1-3 of FCI recursively
                    r1_add = self._apply_rule1(graph, u, a, c)
                    r2_add = self._apply_rule2(graph, u, a, c)
                    r3_add = self._apply_rule3(graph, u, a, c)

                    # apply R4, orienting discriminating paths
                    r4_add, _ = self._apply_rule4(graph, u, a, c, sep_set)

                    # apply R8 to orient more tails
                    r8_add = self._apply_rule8(graph, u, a, c)

                    # apply R9-10 to orient uncovered potentially directed paths
                    r9_add, _ = self._apply_rule9(graph, u, a, c)

                    # a and c are neighbors of u, so u is the endpoint
                    # desired
                    r10_add, _ = self._apply_rule10(graph, a, c, u)

                    # see if there was a change flag
                    if (
                        any(
                            [
                                r1_add,
                                r2_add,
                                r3_add,
                                r4_add,
                                r8_add,
                                r9_add,
                                # r10_add
                            ]
                        )
                        and not change_flag
                    ):
                        logger.debug("Got here...")
                        logger.debug([r1_add, r2_add, r3_add, r4_add, r8_add])
                        logger.debug(change_flag)
                        change_flag = True

            # check if we should continue or not
            if not change_flag:
                # print(r1_add, r2_add, r3_add)
                finished = True
                logger.debug(f"Finished applying R1-4, and R8-9 with {idx} iterations")
                break
            idx += 1

    def _learn_better_skeleton(
        self, X, pag: nx.Graph, sep_set: Dict[str, Dict[str, Set[Any]]], fixed_edges: Set = set()
    ):
        from causal_networkx.discovery.skeleton import learn_skeleton_graph

        adj_graph = pag.to_adjacency_graph()

        # perform pairwise tests to learn skeleton
        skel_graph, sep_set = learn_skeleton_graph(
            X,
            adj_graph,
            fixed_edges,
            sep_set,
            self.ci_estimator,
            self.alpha,
            min_cond_set_size=1,
            max_cond_set_size=self.max_cond_set_size,
            only_neighbors=False,
            max_path_length=self.max_path_length,
            pag=pag,
            **self.ci_estimator_kwargs,
        )
        return skel_graph, sep_set

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
        # initialize the graph
        graph, sep_set, fixed_edges = self._initialize_graph(X)

        # learn the skeleton of the graph
        skel_graph, sep_set = self._learn_skeleton_from_neighbors(X, graph, sep_set, fixed_edges)

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = PAG(incoming_uncertain_data=skel_graph, name="PAG derived with FCI")

        # orient colliders
        self._orient_colliders(pag, sep_set)

        # # now compute all possibly d-separating sets and learn a better skeleton
        skel_graph, sep_set = self._learn_better_skeleton(X, pag, sep_set, fixed_edges)
        return skel_graph, sep_set

    def fit(self, X: pd.DataFrame):
        """Perform causal discovery algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset.

        Returns
        -------
        self : instance of FCI
            FCI instance with fitted attributes.
        """
        # learn skeleton
        skel_graph, sep_set = self.learn_skeleton(X)

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = PAG(incoming_uncertain_data=skel_graph, name="PAG derived with FCI")

        # orient colliders again
        self._orient_colliders(pag, sep_set)
        self.orient_coll_graph = pag.copy()

        # run the rest of the rules to orient as many edges
        # as possible
        self._apply_rules_1to10(pag, sep_set)

        self.skel_graph = skel_graph
        self.graph_ = pag
        return self
