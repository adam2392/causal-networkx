from typing import Callable, Union, Dict, Set
from itertools import combinations, permutations
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd

from causal_networkx import CausalGraph, PAG
from causal_networkx.discovery.classes import ConstraintDiscovery
from causal_networkx.discovery.skeleton import learn_skeleton_graph


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

    def _orient_colliders(self, graph: PAG, sep_set: Dict[Dict[Set]]):
        """Orient colliders given a graph and separation set.

        Parameters
        ----------
        graph : PAG
            The partial ancestral graph (PAG).
        sep_set : Dict[Dict[Set]]
            The separating set between any two nodes.
        """
        # for every node in the PAG, evaluate neighbors to
        for u in graph.nodes:
            for v_i, v_j in combinations(graph.neighbors(u), 2):
                # Check that there is no edge between
                # v_i and v_j, else this is a "shielded" collider.
                # Then check to see if 'u' is in the separating
                # set. If it is not, then there is a collider.
                if not self.graph.has_edge(v_i, v_j) and u not in sep_set[v_i][v_j]:
                    self.graph.orient_edge(v_i, u, "arrow")
                    self.graph.orient_edge(v_j, u, "arrow")
                else:
                    # definite non-collider
                    test = 1

    def _apply_rule1(self, graph: PAG, u, a, c) -> bool:
        """Apply rule 1 of the FCI algorithm.

        If A *-> u o-o C, A and C are not adjacent,
        then we can orient the triple as A o-> u -> C.

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
            # check a *-> u o-o c
            if (
                graph.edge_type(a, u) == "arrow"
                and graph.edge_type(u, c) == "circle"
                and graph.edge_type(c, u) == "circle"
            ):
                # orient the edge from u to c and delete
                # the edge from c to u
                graph.orient_edge(u, c, "arrow")
                graph.remove_edge(c, u)
                added_arrows = True

        return added_arrows

    def _apply_rule2(self, graph: PAG, u, a, c):
        """Apply rule 2 of FCI algorithm.

        If A -> u *-> C, or A *-> u -> C, and A o-o C, then
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
        if graph.has_edge(a, c, "circle"):
            # check that a -> u *-> c
            condition_one = (
                graph.edge_type(a, u) == "arrow"
                and graph.edge_type(u, c) == "arrow"
                and not graph.has_edge(u, a)
            )

            # check that a *-> u -> c
            condition_two = (
                graph.edge_type(a, u) == "arrow"
                and graph.edge_type(u, c) == "arrow"
                and not graph.has_edge(c, u)
            )
            if condition_one or condition_two:
                # orient a *-> c
                graph.orient_edge(a, c, "arrow")
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
            condition_one = graph.has_edge(a, u, "arrow") and graph.has_edge(c, u, "arrow")
            if not condition_one:  # add quick check here to skip non-relevant u nodes
                return added_arrows

            # check for all other neighbors to find a
            for v in graph.neighbors(u):
                # check that v is not a, or c
                if v in (a, c):
                    continue

                # check that v *-o u is in the edge set
                if not graph.has_edge(v, u, "circle"):
                    continue

                # check that a *-o v o-* c
                condition_two = graph.has_edge(a, v, "circle") and graph.has_edge(c, v, "circle")
                if condition_one and condition_two:
                    graph.orient_edge(v, u, "arrow")
                    added_arrows = True
        return added_arrows

    def _apply_rule4(self, graph: PAG, sep_set):
        """Apply rule 4 of FCI algorithm.

        If a path, U = <v, ..., a, u, c> is a discriminating
        path between v and c for u, u o-* c, u in SepSet(v, c),
        orient u -> c. Else, orient a <-> u <-> c.

        Parameters
        ----------
        graph : PAG
            _description_
        """
        path = []

        for u in graph.nodes:
            for (a, c) in combinations(graph.neighbors(u), 2):
                if not graph.has_edge(c, a):
                    continue

                if not graph.has_edge(u, c, "arrow"):
                    continue

                # now look for a discriminating path between v and c for  u.

                # look for parents of node  'a'

                # verify that nodes point into 'a'

                # check if node was already explored

                # check if it is a definite collider

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

    def _apply_rules_1to3(self, graph: PAG):
        while 1:
            for u in graph.nodes():
                for (a, c) in combinations(graph.neighbors(u), 2):
                    # apply R1-3 of FCI recursively
                    r1_add = self._apply_rule1(graph, u, a, c)
                    r2_add = self._apply_rule2(graph, u, a, c)
                    r3_add = self._apply_rule3(graph, u, a, c)

        # check if there is a directed path from A to B and an edge
        # between A, B, then can orient A into B

        # otherwise, if B is a collider in <A,B,C> triplet,
        # B is adjacent to D and D is in the separating set between
        # A and C, then can orient D into B.

        # If U is a definite discriminating path between A and B for M,
        # P and R are adjacent to M on the path U, and P-M-R is
        # a triangle, then

    def fit(self, X: pd.DataFrame):
        # learn the skeleton of the graph
        skel_graph, sep_set = self._learn_skeleton(X)

        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint

        # orient colliders

        # run the rest of the rules to orient as many edges
        # as possible

        nodes = []
        for i in range(dataset.shape[1]):
            node = GraphNode(f"X{i + 1}")
            node.add_attribute("id", i)
            nodes.append(node)

        # reorient all edges with CIRCLE Endpoint
        ori_edges = graph.get_graph_edges()
        for ori_edge in ori_edges:
            graph.remove_edge(ori_edge)
            ori_edge.set_endpoint1(Endpoint.CIRCLE)
            ori_edge.set_endpoint2(Endpoint.CIRCLE)
            graph.add_edge(ori_edge)

        sp = SepsetsPossibleDsep(
            dataset,
            graph,
            independence_test_method,
            alpha,
            background_knowledge,
            depth,
            max_path_length,
            verbose,
            cache_variables_map=cache_variables_map,
        )

        rule0(graph, nodes, sep_sets, background_knowledge, verbose)

        waiting_to_deleted_edges = []

        for edge in graph.get_graph_edges():
            node_x = edge.get_node1()
            node_y = edge.get_node2()

            sep_set = sp.get_sep_set(node_x, node_y)

            if sep_set is not None:
                waiting_to_deleted_edges.append((node_x, node_y, sep_set))

        for waiting_to_deleted_edge in waiting_to_deleted_edges:
            dedge_node_x, dedge_node_y, dedge_sep_set = waiting_to_deleted_edge
            graph.remove_edge(graph.get_edge(dedge_node_x, dedge_node_y))
            sep_sets[(graph.node_map[dedge_node_x], graph.node_map[dedge_node_y])] = dedge_sep_set

            if verbose:
                message = (
                    "Possible DSEP Removed "
                    + dedge_node_x.get_name()
                    + " --- "
                    + dedge_node_y.get_name()
                    + " sepset = ["
                )
                for ss in dedge_sep_set:
                    message += graph.nodes[ss].get_name() + " "
                message += "]"
                print(message)

        reorientAllWith(graph, Endpoint.CIRCLE)
        rule0(graph, nodes, sep_sets, background_knowledge, verbose)

        change_flag = True
        first_time = True

        while change_flag:
            change_flag = False
            change_flag = rulesR1R2cycle(graph, background_knowledge, change_flag, verbose)
            change_flag = ruleR3(graph, sep_sets, background_knowledge, change_flag, verbose)

            if change_flag or (
                first_time
                and background_knowledge is not None
                and len(background_knowledge.forbidden_rules_specs) > 0
                and len(background_knowledge.required_rules_specs) > 0
                and len(background_knowledge.tier_map.keys()) > 0
            ):
                change_flag = ruleR4B(
                    graph,
                    max_path_length,
                    dataset,
                    independence_test_method,
                    alpha,
                    sep_sets,
                    change_flag,
                    background_knowledge,
                    cache_variables_map,
                    verbose,
                )

                first_time = False

                if verbose:
                    print("Epoch")

        graph.set_pag(True)

        edges = get_color_edges(graph)

        return graph, edges
