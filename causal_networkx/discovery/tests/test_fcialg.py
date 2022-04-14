import networkx as nx
import numpy as np
import pandas as pd
import pytest

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

    def test_fci_rule1(self):
        # If A *-> u o-o C, A and C are not adjacent,
        # then we can orient the triple as A *-> u -> C.
        G = PAG()
        G.add_edge("A", "u")
        G.add_circle_edge("u", "C", bidirected=True)
        G_copy = G.copy()

        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C")
        assert not G.has_edge("C", "u")
        assert not G.has_edge("u", "A")

        G = G_copy.copy()
        G.add_edge("u", "A")
        self.alg._apply_rule1(G, "u", "A", "C")
        assert G.has_edge("u", "C")
        assert not G.has_edge("C", "u")
        assert G.has_edge("u", "A")

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

        # 2. Test not-added case
        # first test that can't be A <-> u <-> C
        G = G_copy.copy()
        G.remove_edge('A', 'u')
        G.add_bidirected_edge('u', 'A')
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_circle_edge("A", "C")
        assert not added_arrows

        # 3. then test that A <-> u -> C with A o-o C
        G.remove_bidirected_edge("C", "u")
        G.add_edge('u', 'C')
        added_arrows = self.alg._apply_rule2(G, "u", "A", "C")
        assert G.has_edge("A", "C")
        assert G.has_circle_edge('C', "A")
        assert added_arrows

    def test_fci_rule3(self):
        # If A *-> u <-* C, A *-o v o-* C, A/C are not adjacent,
        # and v *-o u, then orient v *-> u.
        G = PAG()

        # start by considering all stars to be empty for A, C, u
        G.add_edge('A', 'u')
        G.add_edge('C', 'u')

        # then consider all circles as bidirected
        G.add_circle_edge('A', 'v', bidirected=True)
        G.add_circle_edge('C', 'v', bidirected=True)
        G.add_circle_edge('v', 'u', bidirected=True)
        G_copy = G.copy()

        self.alg._apply_rule3(G, "u", "A", "C")
        for edge in G_copy.edges:
            assert G.has_edge(*edge)
        for edge in G_copy.circle_edges:
            if edge != ('v', 'u'):
                assert G.has_circle_edge(*edge)
            else:
                assert not G.has_circle_edge(*edge)
        assert G.has_edge('v', 'u')
        
    def test_fci_rule4(self):
        """Test orienting a discriminating path.

        A discriminating path, p, between X and Y is one where:
        - p has at least 3 edges
        - u is non-endpoint and u is adjacent to c
        - v is not adjacent to c
        - every vertex between v and u is a collider on p and parent of c

        <v,..., w, u, c>
        """
        G = PAG()

        G.add_circle_edge('u', 'c', bidirected=True)
        G.add_bidirected_edge('a', 'u')
        sep_set = set()

        # initial test should not add any arrows, since there are only 2 edges
        added_arrows, explored_nodes = self.alg._apply_rule4(G, 'u', 'a', 'c', sep_set)
        assert not added_arrows
        assert explored_nodes == dict()

        # now add another variable, but since a is not a parent of c
        # this is still not a discriminating path
        G.add_bidirected_edge('b', 'a')
        added_arrows, explored_nodes = self.alg._apply_rule4(G, 'u', 'a', 'c', sep_set)
        assert not added_arrows
        assert explored_nodes == dict()

        # add the arrow from a -> c
        G.add_edge('a', 'c')
        print(G.edges)
        print(G.bidirected_edges)
        print(G.circle_edges)
        added_arrows, explored_nodes = self.alg._apply_rule4(G, 'u', 'a', 'c', sep_set)
        print(G.edges)
        print(G.bidirected_edges)
        print(G.circle_edges)
        assert added_arrows
        assert list(explored_nodes.keys()) == ['c', 'u', 'a', 'b']

        # create a discriminating path

        pass

    def test_fci_skel_graph(self):
        sample = self.scm.sample(n=1, include_latents=False)
        skel_graph, _ = self.alg._learn_skeleton(sample)
        assert list(skel_graph.edges) == [("x", "y"), ("z", "y")]

    def test_fci_basic_collider(self):
        sample = self.scm.sample(n=1, include_latents=False)
        skel_graph, sep_set = self.alg._learn_skeleton(sample)
        graph = PAG(incoming_uncertain_data=skel_graph)
        self.alg._orient_colliders(graph, sep_set)

        # the PAG learned
        expected_graph = PAG()
        expected_graph.add_edges_from([("x", "y"), ("z", "y")])
        expected_graph.add_circle_edges_from([("y", "x"), ("y", "z")])
        assert set(expected_graph.edges) == set(graph.edges)
        assert set(expected_graph.circle_edges) == set(graph.circle_edges)

    def test_fci_unobserved_confounder(self):
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

        expected_pag = PAG()
        expected_pag.add_bidirected_edge("x1", "x2")
        expected_pag.add_edges_from([("x4", "x2"), ("x3", "x1")])
        
        expected_pag_digraph = expected_pag.compute_full_graph(to_networkx=True)
        pag_digraph = pag.compute_full_graph(to_networkx=True)
        print(pag.edges)
        print(pag.circle_edges)
        print(pag.bidirected_edges)
        
        print(expected_pag.edges)
        print(expected_pag.circle_edges)
        print(expected_pag.bidirected_edges)
        
        assert nx.is_isomorphic(pag_digraph, expected_pag_digraph)

    def test_fci_complex(self):
        """
        Test FCI algorithm with more complex graph.

        Use Figure 2 from [1].

        References
        ----------
        [1] https://arxiv.org/pdf/1104.5617.pdf
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

        expected_pag = PAG()
        expected_pag.add_circle_edges_from(
            [("x6", "x5"), ("x2", "x3"), ("x4", "x3"), ("x6", "x4")]
        )
        expected_pag.add_edges_from(
            [
                ('x4', 'x1'), ('x2', 'x5'), ('x3', 'x2'), ('x3', 'x4'),
                ('x2', 'x6'), ('x3', 'x6'), ('x4', 'x6'), ('x5', 'x6')
            ]
        )
        expected_pag.add_bidirected_edge("x1", "x2")
        expected_pag.add_bidirected_edge("x4", "x5")

        for dat in pag.edges.data():
            print(dat)
        print("\n\n now expected...")
        for dat in expected_pag.edges.data():
            print(dat)
        # assert nx.is_isomorphic(pag, expected_pag)

        assert set(pag.circle_edges) == set(expected_pag.circle_edges)
        assert set(pag.edges) == set(expected_pag.edges)
        assert set(pag.bidirected_edges) == set(expected_pag.bidirected_edges)
