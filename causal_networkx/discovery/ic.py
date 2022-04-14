import logging
from collections import deque
from itertools import chain, combinations
from typing import Callable, Dict, Set, Union
from warnings import warn

import networkx as nx
import numpy as np
import pandas as pd

from causal_networkx import PAG, CausalGraph
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
        [1] Jiji Zhang. On the completeness of orientation rules
        for causal discovery in the presence of latent confounders
        and selection bias. Artificial Intelligence,
        172(16):1873–1896, 2008.

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
                    logger.debug(f"orienting collider: {v_i} -> {u} and {v_j} -> {u}")

                    if graph.has_circle_edge(v_i, u):
                        graph.orient_circle_edge(v_i, u, "arrow")
                    if graph.has_circle_edge(v_j, u):
                        graph.orient_circle_edge(v_j, u, "arrow")
                # else:
                # definite non-collider
                # test = 1

    def _apply_rule1(self, graph: PAG, u, a, c) -> bool:
        """Apply rule 1 of the FCI algorithm.

        If A *-> u o-o C, A and C are not adjacent,
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
        # check that a and c are not adjacent
        if not graph.has_adjacency(a, c):  # has_edge(a, c) and not graph.has_edge(c, a):
            # check a *-> u o-o c
            if (
                graph.edge_type(a, u) == "arrow"
                and graph.edge_type(u, c) == "circle"
                and graph.edge_type(c, u) == "circle"
            ):
                logger.debug(f"Rule 1: Orienting edge {u} o-o {c} to {u} -> {c}.")
                # orient the edge from u to c and delete
                # the edge from c to u
                graph.orient_circle_edge(u, c, "arrow")
                graph.orient_circle_edge(c, u, "tail")
                added_arrows = True

        return added_arrows

    def _apply_rule2(self, graph: PAG, u, a, c):
        """Apply rule 2 of FCI algorithm.

        If A -> u *-> C, or A *-> u -> C, and A *-o C, then
        orient A *-> C.

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
            # check that a -> u *-> c
            condition_one = (
                graph.edge_type(a, u) == "arrow"
                and graph.edge_type(u, c) in ["arrow", "bidirected"]
                and not graph.has_edge(u, a)
            )

            # check that a *-> u -> c
            condition_two = (
                graph.edge_type(a, u) in ["arrow", "bidirected"]
                and graph.edge_type(u, c) == "arrow"
                and not graph.has_edge(c, u)
            )

            if condition_one or condition_two:
                logger.debug(f"Rule 2: Orienting circle edge to {a} -> {c}")
                # orient a *-> c
                graph.orient_circle_edge(a, c, "arrow")
                added_arrows = True
        return added_arrows

    def _apply_rule3(self, graph: PAG, u, a, c):
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
        if not graph.has_edge(a, c) and not graph.has_edge(c, a):
            # check that a *-> u <-* c
            condition_one = graph.has_edge(a, u) and graph.has_edge(c, u)
            if not condition_one:  # add quick check here to skip non-relevant u nodes
                return added_arrows

            # check for all other neighbors to find a
            for v in graph.neighbors(u):
                # check that v is not a, or c
                if v in (a, c):
                    continue

                # check that v *-o u is in the edge set
                if not graph.has_circle_edge(v, u):
                    continue

                # check that a *-o v o-* c
                condition_two = graph.has_circle_edge(a, v) and graph.has_circle_edge(c, v)
                if condition_one and condition_two:
                    graph.orient_circle_edge(v, u, "arrow")
                    added_arrows = True
        return added_arrows

    def _apply_rule4(self, graph: PAG, u, a, c, sep_set):
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
        """
        added_arrows = False
        explored_nodes = dict()

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

        # parents of c form the discriminating path
        cparents = graph.parents(c)

        # keep track of the distance searched
        distance = 0

        # keep track of the previous nodes, i.e. to build a path
        # from node (key) to its child along the path (value)
        descendant_nodes = dict()
        descendant_nodes[c] = u

        # keep track of paths of certain nodes that were already explored
        explored_nodes[c] = None
        explored_nodes[u] = None
        explored_nodes[a] = None

        # start off with the valid triple <a, u, c>
        # - u is adjacent to c
        # - u has an arrow pointing to a
        # - TBD a is a definite collider
        # - TBD endpoint is not adjacent to c
        explored_list = [c, u, a]

        # now add 'a' to the queue and begin exploring
        # adjacent nodes that are connected with bidirected edges
        path = deque([a])
        while not len(path) == 0:
            this_node = path.popleft()

            # check distance criterion to prevent checking very long paths
            distance += 1
            print(self.max_path_length, distance)
            if distance > 0 and distance > (
                1000 if self.max_path_length == np.inf else self.max_path_length
            ):
                warn(f'Did not finish checking discriminating path in {self} because the path '
                     f'length exceeded {self.max_path_length}.')
                return added_arrows, explored_nodes

            # now we check all neighbors to this_node that are pointing to it
            # either with a direct edge, or a bidirected edge
            for next_node in chain(
                graph.parents(this_node), graph.c_component_graph.neighbors(this_node)
            ):

                # now check all bidirected connections with this_node
                # for next_node in graph.c_component_graph.neighbors(this_node):
                # if we have already explored this neighbor, then it is
                # already along the potentially discriminating path
                if next_node in explored_nodes:
                    continue

                # This node is a definite collider since there was
                # confirmed an arrow pointing towards 'this_node'
                # and 'next_node' is connected to it via a bidirected arrow.
                # Check if it is now the end of the discriminating path.
                # Note we now have 3 edges in the path by construction.
                if not graph.has_adjacency(next_node, c) and next_node != c:
                    logger.debug(f"Reached the end of the discriminating path with {next_node}.")

                    # now check if u is in SepSet(v, c)
                    # handle edge case where sep_set is empty.
                    if next_node in sep_set:
                        if u in sep_set[next_node][c]:
                            # orient u -> c
                            graph.remove_edge(c, u)
                            graph.orient_circle_edge(u, c, "arrow")
                    else:
                        # orient u <-> c
                        if graph.has_circle_edge(u, c):
                            graph.orient_circle_edge(u, c, "arrow")
                        if graph.has_circle_edge(c, u):
                            graph.orient_circle_edge(c, u, "arrow")
                    added_arrows = True
                    explored_list.append(next_node)
                    explored_nodes[next_node] = None
                    break

                # If we didn't reach the end of the discriminating path,
                # then we can add 'next_node' to the path. This only occurs
                # if 'next_node' is a valid new entry, which requires it
                # to be a part of the parents of 'c'.
                if next_node in cparents:
                    # since it is a parent, we can now add it to the path queue
                    path.append(next_node)
                    explored_list.append(next_node)
                    explored_nodes[next_node] = None

        # now look for a discriminating path between v and c for u.
        # path = deque([c])
        # while not len(path) == 0:
        #     # get the current node to evaluate
        #     node_i = path.popleft()

        #     # check distance criterion to prevent checking very long paths
        #     distance += 1
        #     if distance > 0 and distance > (
        #         1000 if self.max_path_length == np.inf else self.max_path_length
        #     ):
        #         return added_arrows, explored_nodes

        #     # check all bidirected

        #     # check all parents of node_i to see if we can find
        #     # a discriminating path
        #     node_i_pa = graph.parents(node_i)
        #     for node_next in node_i_pa:
        #         if node_next in explored_nodes:
        #             continue

        #         descendant_nodes[node_next] = node_i
        #         node_i_child = descendant_nodes[node_i]

        #         # check that the next node is a definite collider
        #         if not graph.is_def_collider(node_next, node_i, node_i_child):
        #             print('Next node is not a definite collider', node_next)
        #             continue

        #         # all nodes along a disc. path must be parent of 'c'
        #         # else it is the end of a discriminating path
        #         if not graph.has_adjacency(node_next, c) and node_next != c:
        #             logger.debug(f'Reached the end of the discriminating path with {node_next}.')

        #             # now check if u is in SepSet(v, c)
        #             if u in sep_set[node_next][c]:
        #                 # orient u -> c
        #                 graph.remove_edge(c, u)
        #                 graph.orient_circle_edge(u, c, "arrow")
        #             else:
        #                 # orient u <-> c
        #                 graph.orient_circle_edge(u, c, "arrow")
        #                 graph.orient_circle_edge(c, u, "arrow")
        #             added_arrows = True
        #             break

        #         # update 'c' parents
        #         if node_next in cparents:
        #             path.append(node_next)
        #             explored_nodes[node_next] = None
        return added_arrows, explored_nodes

    def _apply_rule5(self, graph: PAG):
        pass

    def _apply_rule6(self, graph: PAG):
        pass

    def _apply_rule7(self, graph: PAG):
        pass

    def _apply_rule8(self, graph: PAG):
        pass

    def _apply_rule9(self, graph: PAG):
        pass

    def _apply_rule10(self, graph: PAG):
        pass

    def _apply_rules_1to3(self, graph: PAG, sep_set: Set):
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:
            for u in graph.nodes():
                for (a, c) in combinations(graph.neighbors(u), 2):
                    # apply R1-3 of FCI recursively
                    r1_add = self._apply_rule1(graph, u, a, c)
                    r2_add = self._apply_rule2(graph, u, a, c)
                    r3_add = self._apply_rule3(graph, u, a, c)

                    change_flag = any([r1_add, r2_add, r3_add])

                    # apply R4, orienting discriminating paths
                    r4_add, _ = self._apply_rule4(graph, u, a, c, sep_set)

                    # check if we should continue or not
                    if not all(added_edges for added_edges in [r1_add, r2_add, r3_add, r4_add]):
                        finished = True
                        break
            idx += 1

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
        # learn the skeleton of the graph
        skel_graph, sep_set = self._learn_skeleton(X)

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = PAG(incoming_uncertain_data=skel_graph, name="PAG derived with FCI")

        # orient colliders
        self._orient_colliders(pag, sep_set)
        self.orient_coll_graph = pag.copy()

        # run the rest of the rules to orient as many edges
        # as possible
        self._apply_rules_1to3(pag, sep_set)

        # then run rule 4

        self.skel_graph = skel_graph
        self.graph_ = pag
        return self
