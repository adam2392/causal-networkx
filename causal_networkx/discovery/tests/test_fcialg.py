import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_networkx.algorithms import d_separated, possibly_d_sep_sets
from causal_networkx.cgm import PAG, CausalGraph
from causal_networkx.ci import Oracle
from causal_networkx.discovery import FCI
from causal_networkx.scm import StructuralCausalModel


class Test_FCI:
    def setup_method(self):
        seed = 12345
        rng = np.random.RandomState(seed=seed)

        # construct a causal graph that will result in
        # x -> y <- z
        func_uz = lambda: rng.negative_binomial(n=1, p=0.25)
        func_uxy = lambda: rng.binomial(n=1, p=0.4)
        func_x = lambda u_xy: 2 * u_xy
        func_y = lambda x, u_xy, z: x * u_xy + z
        func_z = lambda u_z: u_z

        # construct the SCM and the corresponding causal graph
        scm = StructuralCausalModel(
            exogenous={
                "u_xy": func_uxy,
                "u_z": func_uz,
            },
            endogenous={"x": func_x, "y": func_y, "z": func_z},
        )
        G = scm.get_causal_graph()
        oracle = Oracle(G)

        self.scm = scm
        self.G = G
        self.ci_estimator = oracle.ci_test

        fci = FCI(ci_estimator=self.ci_estimator)
        self.alg = fci

    def test_fci_skel_graph(self):
        sample = self.scm.sample(n=1, include_latents=False)
        skel_graph, _ = self.alg.learn_skeleton(sample)
        assert list(skel_graph.edges) == [("x", "y"), ("z", "y")]

    def test_fci_basic_collider(self):
        sample = self.scm.sample(n=1, include_latents=False)
        skel_graph, sep_set = self.alg.learn_skeleton(sample)
        graph = PAG(incoming_uncertain_data=skel_graph)
        self.alg._orient_colliders(graph, sep_set)

        # the PAG learned
        expected_graph = PAG()
        expected_graph.add_edges_from([("x", "y"), ("z", "y")])
        expected_graph.add_circle_edges_from([("y", "x"), ("y", "z")])
        assert set(expected_graph.edges) == set(graph.edges)
        assert set(expected_graph.circle_edges) == set(graph.circle_edges)

    def test_fci_rule1(self):
        # If A *-> u o-* C, A and C are not adjacent,
        # then we can orient the triple as A *-> u -> C.

        # First test:
        # A -> u o-o C
        G = PAG()
        G.add_edge("A", "u")
        G.add_circle_edge("u", "C", bidirected=True)
        G_copy = G.copy()

        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C")
        assert not G.has_circle_edge("C", "u")
        assert not G.has_edge("C", "u")
        assert not G.has_edge("u", "A")

        # orient u o-o C now as u o-> C
        # Second test:
        # A -> u o-> C
        G = G_copy.copy()
        G.orient_circle_edge("u", "C", "arrow")
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C")
        assert not G.has_circle_edge("C", "u")
        assert not G.has_edge("C", "u")
        assert not G.has_edge("u", "A")

        # now orient A -> u as A <-> u
        # Third test:
        # A <-> u o-o C
        G = G_copy.copy()
        G.remove_edge("A", "u")
        G.add_bidirected_edge("u", "A")
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C")
        assert not G.has_circle_edge("C", "u")
        assert not G.has_edge("C", "u")
        assert G.has_bidirected_edge("u", "A")

        # now orient A -> u as A <-> u
        # Fourth test:
        # A o-> u o-o C
        G = G_copy.copy()
        G.add_circle_edge("u", "A")
        G.orient_circle_edge("u", "C", "arrow")
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C")
        assert not G.has_circle_edge("C", "u")
        assert not G.has_edge("C", "u")
        assert G.has_circle_edge("u", "A")

    def test_fci_rule2(self):
        # If A -> u *-> C, or A *-> u -> C, and A *-o C, then
        # orient A *-> C.
        # 1. Do A -> u <-> C with A o-o C
        G = PAG()
        G.add_edge("A", "u")
        G.add_bidirected_edge("u", "C")
        G.add_circle_edge("A", "C", bidirected=True)
        G_copy = G.copy()

        self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_edge("A", "C")
        assert G.has_circle_edge("C", "A")

        # if A o-> u, then it should not work
        G = G_copy.copy()
        G.add_circle_edge("u", "A")
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert not added_arrows
        assert G.has_circle_edge("A", "C")
        assert G.has_circle_edge("C", "A")

        # 2. Test not-added case
        # first test that can't be A <-> u <-> C
        G = G_copy.copy()
        G.remove_edge("A", "u")
        G.add_bidirected_edge("u", "A")
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_circle_edge("A", "C")
        assert not added_arrows

        # 3. then test that A <-> u -> C with A o-o C
        G.remove_bidirected_edge("C", "u")
        G.add_edge("u", "C")
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_edge("A", "C")
        assert G.has_circle_edge("C", "A")
        assert added_arrows

    def test_fci_rule3(self):
        # If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
        # and v *-o u, then orient v *-> u.
        G = PAG()

        # start by considering all stars to be empty for A, C, u
        G.add_edge("A", "u")
        G.add_edge("C", "u")

        # then consider all circles as bidirected
        G.add_circle_edge("A", "v", bidirected=True)
        G.add_circle_edge("C", "v", bidirected=True)
        G.add_circle_edge("v", "u", bidirected=True)
        G_copy = G.copy()

        self.alg._apply_rule3(G, "u", "A", "C")
        for edge in G_copy.edges:
            assert G.has_edge(*edge)
        for edge in G_copy.circle_edges:
            if edge != ("v", "u"):
                assert G.has_circle_edge(*edge)
            else:
                assert not G.has_circle_edge(*edge)
        assert G.has_edge("v", "u")

        # if A -> u is A <-> u, then it should still work
        G = G_copy.copy()
        G.remove_edge("A", "u")
        G.add_bidirected_edge("A", "u")
        added_arrows = self.alg._apply_rule3(G, "u", "A", "C")
        assert added_arrows

        # adding a circle edge should make it not work
        G = G_copy.copy()
        G.add_circle_edge("A", "C", bidirected=True)
        added_arrows = self.alg._apply_rule3(G, "u", "A", "C")
        assert not added_arrows

    def test_fci_rule4_without_sepset(self):
        """Test orienting a discriminating path without separating set.

        A discriminating path, p, between X and Y is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        <v,..., w, u, c>
        """
        G = PAG()

        # setup graph with a <-> u o-o c
        G.add_circle_edge("u", "c", bidirected=True)
        G.add_bidirected_edge("a", "u")
        sep_set = set()

        # initial test should not add any arrows, since there are only 2 edges
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == dict()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        # setup graph with b <-> a <-> u o-o c
        G.add_bidirected_edge("b", "a")
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == dict()

        # add the arrow from a -> c
        G.add_edge("a", "c")
        G_copy = G.copy()
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert list(explored_nodes.keys()) == ["c", "u", "a", "b"]

        # since separating set is empty
        assert not G.has_circle_edge("c", "u")
        assert G.has_bidirected_edge("c", "u")

        # change 'u' o-o 'c' to 'u' o-> 'c', which should now orient
        # the same way
        G = G_copy.copy()
        G.orient_circle_edge("u", "c", "arrow")
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert list(explored_nodes.keys()) == ["c", "u", "a", "b"]
        assert not G.has_circle_edge("c", "u")
        assert G.has_bidirected_edge("c", "u")

    def test_fci_rule4_early_exit(self):
        G = PAG()

        G.add_circle_edge("u", "c", bidirected=True)
        G.add_bidirected_edge("a", "u")
        sep_set = set()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        G.add_bidirected_edge("b", "a")
        G.add_edge("a", "c")
        G.add_edge("b", "c")
        G.add_edge("d", "b")

        # test error case
        new_fci = FCI(ci_estimator=self.ci_estimator, max_path_length=1)
        with pytest.warns(UserWarning, match="Did not finish checking"):
            new_fci._apply_rule4(G, "u", "a", "c", sep_set)

    def test_fci_rule4_wit_sepset(self):
        """Test orienting a discriminating path with a separating set.

        A discriminating path, p, between X and Y is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        <v,..., w, u, c>
        """
        G = PAG()

        G.add_circle_edge("u", "c", bidirected=True)
        G.add_bidirected_edge("a", "u")
        sep_set = {"b": {"c": set("u")}}

        # initial test should not add any arrows, since there are only 2 edges
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == dict()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        G.add_bidirected_edge("b", "a")
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert not added_arrows
        assert explored_nodes == dict()

        # add the arrow from a -> c
        G.add_edge("a", "c")
        G_copy = G.copy()
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert list(explored_nodes.keys()) == ["c", "u", "a", "b"]
        assert not G.has_circle_edge("c", "u")
        assert not G.has_edge("c", "u")
        assert G.has_edge("u", "c")

        # change 'u' o-o 'c' to 'u' o-> 'c', which should now orient
        # the same way
        G = G_copy.copy()
        G.orient_circle_edge("u", "c", "arrow")
        added_arrows, explored_nodes = self.alg._apply_rule4(G, "u", "a", "c", sep_set)
        assert added_arrows
        assert list(explored_nodes.keys()) == ["c", "u", "a", "b"]
        assert not G.has_circle_edge("c", "u")
        assert not G.has_edge("c", "u")
        assert G.has_edge("u", "c")

    def test_fci_rule8_without_selection_bias(self):
        # If A -> u -> C and A o-> C
        # orient A o-> C as A -> C
        G = PAG()

        # create a chain for A, u, C
        G.add_chain(["A", "u", "C"])
        G.add_edge("A", "C")
        G.add_circle_edge("C", "A")
        self.alg._apply_rule8(G, "u", "A", "C")

        assert G.has_edge("A", "C")
        assert not G.has_circle_edge("C", "A")

    def test_fci_rule9(self):
        # If A o-> C and there is an undirected pd path
        # from A to C through u, where u and C are not adjacent
        # then orient A o-> C as A -> C
        G = PAG()

        # create an uncovered pd path from A to C through u
        G.add_edge("A", "C")
        G.add_circle_edge("C", "A")
        G.add_chain(["A", "u", "x", "y", "z", "C"])
        G.add_circle_edge("y", "x")

        # create a pd path from A to C through v
        G.add_chain(["A", "v", "x", "y", "z", "C"])
        # with the bidirected edge, v,x,y is a shielded triple
        G.add_bidirected_edge("v", "y")
        G_copy = G.copy()

        # get the uncovered pd paths
        added_arrows, uncov_pd_path = self.alg._apply_rule9(G, "u", "A", "C")
        assert added_arrows
        assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]
        assert not G.has_circle_edge("C", "A")

        # the shielded triple should not result in an uncovered pd path
        G = G_copy.copy()
        added_arrows, uncov_pd_path = self.alg._apply_rule9(G, "v", "A", "C")
        assert not added_arrows
        assert uncov_pd_path == []
        assert G.has_circle_edge("C", "A")

        # when there is a circle edge it should still work
        G = G_copy.copy()
        G.add_circle_edge("C", "z")
        added_arrows, uncov_pd_path = self.alg._apply_rule9(G, "u", "A", "C")
        assert added_arrows
        assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]
        assert not G.has_circle_edge("C", "A")

    def test_fci_rule10(self):
        # If A o-> C and u -> C <- v and:
        # - there is an uncovered pd path from A to u, p1
        # - there is an uncovered pd from from A to v, p2
        # if mu adjacent to A on p1 is distinct from w adjacent to A on p2
        # and mu is not adjacent to w, then orient orient A o-> C as A -> C
        G = PAG()

        # make A o-> C
        G.add_edge("A", "C")
        G.add_circle_edge("C", "A")
        # create an uncovered pd path from A to u that ends at C
        G.add_chain(["A", "x", "y", "z", "u", "C"])
        G.add_circle_edge("y", "x")
        G_copy = G.copy()

        # create an uncovered pd path from A to v so now C is a collider for <u, C, v>
        G.add_chain(["A", "x", "y", "z", "v", "C"])

        # 'x' and 'x' are not distinct, so won't orient
        added_arrows, a_to_u_path, a_to_v_path = self.alg._apply_rule10(G, 'u', 'A', 'C')
        assert not added_arrows
        assert a_to_u_path == []
        assert a_to_v_path == []
        assert G.has_circle_edge('C', 'A')

        # if we create an edge from A -> y, there is now a distinction
        G = G_copy.copy()
        G.add_edge('A', 'y')
        added_arrows, a_to_u_path, a_to_v_path = self.alg._apply_rule10(G, 'u', 'A', 'C')
        assert added_arrows
        assert a_to_u_path == ['A', 'x', 'y', 'z', 'u']
        assert a_to_v_path == ['A', 'y', 'z', 'v']
        
    def test_fci_unobserved_confounder(self):
        # x4 -> x2 <- x1 <- x3
        # x1 <--> x2
        # x4 | x1,
        edge_list = [
            ("x4", "x2"),
            ("x3", "x1"),
            ("x1", "x2"),
        ]
        latent_edge_list = [("x1", "x2")]
        G = CausalGraph(edge_list, latent_edge_list)
        sample = np.random.normal(size=(len(G.nodes), 5)).T
        sample = pd.DataFrame(sample)
        sample.columns = list(G.nodes)

        oracle = Oracle(G)
        ci_estimator = oracle.ci_test
        fci = FCI(ci_estimator=ci_estimator)
        fci.fit(sample)
        pag = fci.graph_

        print(fci.skel_graph.edges)

        expected_pag = PAG()
        expected_pag.add_edges_from(
            [
                ("x4", "x2"),
                ("x1", "x2"),
                ("x3", "x2"),
            ]
        )
        expected_pag.add_circle_edges_from(
            [("x2", "x4"), ("x2", "x3"), ("x2", "x1"), ("x1", "x3"), ("x3", "x1")]
        )

        assert set(pag.edges) == set(expected_pag.edges)
        assert set(pag.bidirected_edges) == set(expected_pag.bidirected_edges)
        assert set(pag.circle_edges) == set(expected_pag.circle_edges)

        expected_pag_digraph = expected_pag.compute_full_graph(to_networkx=True)
        pag_digraph = pag.compute_full_graph(to_networkx=True)
        assert nx.is_isomorphic(pag_digraph, expected_pag_digraph)

    def test_fci_spirtes_example(self):
        """Test example in book.

        See book Figure 16

        See: https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/.g/scottd/fullbook.pdf
        """
        # reconstruct the PAG the way FCI would
        edge_list = [("D", "A"), ("B", "E"), ("F", "B"), ("C", "F"), ("C", "H"), ("H", "D")]
        latent_edge_list = [("A", "B"), ("D", "E")]
        graph = CausalGraph(edge_list, latent_edge_list)
        alg = FCI(ci_estimator=Oracle(graph).ci_test)

        sample = graph.dummy_sample()
        alg.fit(sample)
        pag = alg.graph_

        # generate the expected PAG
        edge_list = [
            ("D", "A"),
            ("B", "E"),
            ("H", "D"),
            ("F", "B"),
        ]
        latent_edge_list = [("A", "B"), ("D", "E")]
        uncertain_edge_list = [
            ("B", "F"),
            ("F", "C"),
            ("C", "F"),
            ("C", "H"),
            ("H", "C"),
            ("D", "H"),
        ]
        expected_pag = PAG(edge_list, latent_edge_list, uncertain_edge_list)

        assert set(expected_pag.bidirected_edges) == set(pag.bidirected_edges)
        assert set(expected_pag.edges) == set(pag.edges)
        assert set(expected_pag.circle_edges) == set(pag.circle_edges)

    def test_fci_complex(self):
        """
        Test FCI algorithm with more complex graph.

        Use Figure 2 from :footcite:`Colombo2012`.

        References
        ----------
        .. footbibliography::
        """
        edge_list = [
            ("x4", "x1"),
            ("x2", "x5"),
            ("x3", "x2"),
            ("x3", "x4"),
            ("x2", "x6"),
            ("x3", "x6"),
            ("x4", "x6"),
            ("x5", "x6"),
        ]
        latent_edge_list = [("x1", "x2"), ("x4", "x5")]
        G = CausalGraph(edge_list, latent_edge_list)
        sample = np.random.normal(size=(len(G.nodes), 5)).T
        sample = pd.DataFrame(sample)
        sample.columns = list(G.nodes)

        oracle = Oracle(G)
        ci_estimator = oracle.ci_test
        fci = FCI(ci_estimator=ci_estimator, max_iter=np.inf)
        fci.fit(sample)
        pag = fci.graph_

        assert d_separated(G, "x1", "x3", "x4")
        pdsep = possibly_d_sep_sets(pag, "x1", "x3")
        assert "x2" in pdsep

        expected_pag = PAG()
        expected_pag.add_circle_edges_from([("x6", "x5"), ("x2", "x3"), ("x4", "x3"), ("x6", "x4")])
        expected_pag.add_edges_from(
            [
                ("x4", "x1"),
                ("x2", "x5"),
                ("x3", "x2"),
                ("x3", "x4"),
                ("x2", "x6"),
                ("x3", "x6"),
                ("x4", "x6"),
                ("x5", "x6"),
            ]
        )
        expected_pag.add_bidirected_edge("x1", "x2")
        expected_pag.add_bidirected_edge("x4", "x5")

        print(pag.to_adjacency_graph().edges)
        print(pag.edges)
        print(pag.bidirected_edges)
        print(pag.circle_edges)
        assert set(pag.bidirected_edges) == set(expected_pag.bidirected_edges)
        assert set(pag.edges) == set(expected_pag.edges)
        assert set(pag.circle_edges) == set(expected_pag.circle_edges)
