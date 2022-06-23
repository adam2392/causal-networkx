import logging
from collections import defaultdict
from itertools import combinations, permutations
from typing import Dict, List, Set, Union

import networkx as nx

from causal_networkx import ADMG, CPDAG, DAG
from causal_networkx.ci.base import BaseConditionalIndependenceTest
from causal_networkx.graphs.intervention import PsiCPDAG, PsiPAG

from .fcialg import FCI
from .pcalg import PC

logger = logging.getLogger()


class PsiPC(PC):
    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
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
        super().__init__(
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size,
            max_cond_set_size,
            max_iter,
            max_combinations,
            apply_orientations,
            **ci_estimator_kwargs,
        )

    def convert_skeleton_graph(self, graph: nx.Graph) -> PsiCPDAG:
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
        # convert Graph object to a PsiCPDAG object with
        # all undirected edges
        graph = PsiCPDAG(incoming_uncertain_data=graph)
        return graph

    def orient_edges(self, skel_graph: CPDAG, sep_set: Dict[str, Dict[str, Set]]) -> CPDAG:
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
                r1_add = self._apply_meek_rule1(skel_graph, i, j)

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                r2_add = self._apply_meek_rule2(skel_graph, i, j)

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                r3_add = self._apply_meek_rule3(skel_graph, i, j)

                r4_add = self._apply_psi_rule1(skel_graph, i, j)
                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent.
                #
                # However, this rule is not necessary when the PC-algorithm
                # is used to estimate a DAG.

                if any([r1_add, r2_add, r3_add, r4_add]) and not change_flag:
                    change_flag = True
            if not change_flag:
                finished = True
                logger.info(f"Finished applying R1-3, with {idx} iterations")
                break
            idx += 1

        return skel_graph

    def _apply_psi_rule1(self, skel_graph, i, j):
        pass


class PsiFCI(FCI):
    """Psi FCI algorithm for learning an interventional PAG.

    Parameters
    ----------
    ci_estimator : Callable
        The conditional independence test function. The arguments of the estimator should
        be data, node, node to compare, conditioning set of nodes, and any additional
        keyword arguments.
    alpha : float, optional
        The significance level for the conditional independence test, by default 0.05.
    init_graph : nx.Graph
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
        apply Zhang's orientation rules R0-10, orienting colliders and certain
        arrowheads and tails :footcite:`Zhang2008`.
    selection_bias : bool
        Whether or not to account for selection bias within the causal PAG.
        See :footcite:`Zhang2008`. Not used.
    max_path_length : int, optional
        The maximum length of any discriminating path, or None if unlimited.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        init_graph: Union[nx.Graph, DAG, ADMG] = None,
        fixed_edges: nx.Graph = None,
        min_cond_set_size: int = None,
        max_cond_set_size: int = None,
        max_iter: int = 1000,
        max_combinations: int = None,
        apply_orientations: bool = True,
        selection_bias: bool = False,
        max_path_length: int = None,
        **ci_estimator_kwargs,
    ):
        super().__init__(
            ci_estimator,
            alpha,
            init_graph,
            fixed_edges,
            min_cond_set_size,
            max_cond_set_size,
            max_iter,
            max_combinations,
            apply_orientations,
            selection_bias,
            max_path_length,
            **ci_estimator_kwargs,
        )

    def convert_skeleton_graph(self, graph: nx.Graph) -> PsiPAG:
        # convert the undirected skeleton graph to a PAG, where
        # all left-over edges have a "circle" endpoint
        pag = PsiPAG(incoming_uncertain_data=graph, name="PsiPAG derived with FCI")
        return pag

    def _compute_augmented_nodes(self, intervention_set: List[Set]):
        """Compute the augmented nodes to add to the graph.

        Parameters
        ----------
        intervention_set : List[Set]
            The set of "known" interventions carried out.

        Returns
        -------
        augmented_node_set : List[Set]
            The list of sets of augmented nodes. Each element is a set of nodes
            that corresponds to an F-node.
        sigma_set : List
            The set of interventions corresponding to each augmented node.
        """
        augmented_node_set: Dict[Set, None] = dict()
        sigma_set = []
        idx = 0

        for int_i, int_j in combinations(intervention_set, 2):
            # compute the symmetric difference in the sets
            symm_diff = int_i.symmetric_difference(int_j)

            if symm_diff not in augmented_node_set:
                # add to the augmented node set
                augmented_node_set[symm_diff] = None
                sigma_set.append((int_i, int_j))
                idx += 1
        return augmented_node_set, sigma_set

    def _initialize_graph(self, nodes):
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
            for (node_i, node_j) in combinations(*graph.nodes):
                if not graph.has_edge(node_i, node_j):
                    sep_set[node_i][node_j] = set()
                    sep_set[node_j][node_i] = set()
        return super()._initialize_graph(nodes)

    def _apply_psi_rule1(skel_graph, i, j):
        pass

    def orient_edges(self, graph, sep_set):
        # orient colliders again
        self._orient_colliders(graph, sep_set)
        self.orient_coll_graph = graph.copy()

        # run the rest of the rules to orient as many edges
        # as possible
        idx = 0
        finished = False
        while idx < self.max_iter and not finished:
            change_flag = False
            logger.info(f"Running R1-10 for iteration {idx}")

            for u in graph.nodes:
                for (a, c) in permutations(graph.adjacencies(u), 2):
                    # apply R1-3 to orient triples and arrowheads
                    r1_add = self._apply_rule1(graph, u, a, c)
                    r2_add = self._apply_rule2(graph, u, a, c)
                    r3_add = self._apply_rule3(graph, u, a, c)

                    # apply R4, orienting discriminating paths
                    r4_add, _ = self._apply_rule4(graph, u, a, c, sep_set)

                    # apply R8 to orient more tails
                    r8_add = self._apply_rule8(graph, u, a, c)

                    # apply R9-10 to orient uncovered potentially directed paths
                    r9_add, _ = self._apply_rule9(graph, u, a, c)

                    # a and c are neighbors of u, so u is the endpoint desired
                    r10_add, _, _ = self._apply_rule10(graph, a, c, u)

                    # see if there was a change flag
                    if (
                        any([r1_add, r2_add, r3_add, r4_add, r8_add, r9_add, r10_add])
                        and not change_flag
                    ):
                        logger.info("Got here...")
                        logger.info([r1_add, r2_add, r3_add, r4_add, r8_add, r9_add, r10_add])
                        logger.info(change_flag)
                        change_flag = True

            # check if we should continue or not
            if not change_flag:
                finished = True
                logger.info(f"Finished applying R1-4, and R8-10 with {idx} iterations")
                break
            idx += 1
        return graph
