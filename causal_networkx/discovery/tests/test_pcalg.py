import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_networkx import CPDAG, StructuralCausalModel
from causal_networkx.ci import Oracle, g_square_binary, g_square_discrete
from causal_networkx.ci.tests.testdata import bin_data, dis_data
from causal_networkx.discovery import PC


@pytest.mark.parametrize(
    ("indep_test_func", "data_matrix", "g_answer"),
    [
        (
            g_square_binary,
            np.array(bin_data).reshape((5000, 5)),
            nx.DiGraph(
                {
                    0: (1,),
                    1: (),
                    2: (3, 4),
                    3: (1, 2),
                    4: (1, 2),
                }
            ),
        ),
        (
            g_square_discrete,
            np.array(dis_data).reshape((10000, 5)),
            nx.DiGraph(
                {
                    0: (2,),
                    1: (2, 3),
                    2: (),
                    3: (),
                    4: (3,),
                }
            ),
        ),
    ],
)
def test_estimate_cpdag(indep_test_func, data_matrix, g_answer, alpha=0.01):
    """Test PC algorithm for estimating the causal DAG."""
    data_df = pd.DataFrame(data_matrix)
    alg = PC(ci_estimator=indep_test_func, alpha=alpha)
    alg.fit(data_df)
    graph = alg.graph_

    error_msg = "True edges should be: %s" % (g_answer.edges(),)
    print(graph.all_edges())
    assert nx.is_isomorphic(graph.to_directed(), g_answer), error_msg

    # test what happens if fixed edges are present
    fixed_edges = nx.complete_graph(data_df.columns.values)
    alg = PC(fixed_edges=fixed_edges, ci_estimator=indep_test_func, alpha=alpha)
    alg.fit(data_df)
    complete_graph = alg.graph_
    assert nx.is_isomorphic(complete_graph.to_undirected(), fixed_edges)
    assert not nx.is_isomorphic(complete_graph, g_answer)


def test_collider():
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
    ci_estimator = oracle.ci_test
    pc = PC(ci_estimator=ci_estimator)

    sample = scm.sample(n=1, include_latents=False)
    pc.fit(sample)
    graph = pc.graph_

    assert graph.has_edge("x", "y")
    assert graph.has_edge("z", "y")


class Test_PC:
    def setup_method(self):
        seed = 12345
        rng = np.random.RandomState(seed=seed)

        # construct a causal graph that will result in
        # x -> y <- z
        func_uz = lambda: rng.negative_binomial(n=1, p=0.25)
        func_ux = lambda: rng.binomial(n=1, p=0.4)
        func_x = lambda u_x: 2 * u_x
        func_y = lambda x, z: x + z
        func_z = lambda u_z: u_z

        # construct the SCM and the corresponding causal graph
        scm = StructuralCausalModel(
            exogenous={
                "u_x": func_ux,
                "u_z": func_uz,
            },
            endogenous={"x": func_x, "y": func_y, "z": func_z},
        )
        G = scm.get_causal_graph()
        oracle = Oracle(G)

        self.scm = scm
        self.G = G
        self.ci_estimator = oracle.ci_test
        pc = PC(ci_estimator=self.ci_estimator)
        self.alg = pc

    def test_pc_skel_graph(self):
        sample = self.scm.sample(n=1, include_latents=False)
        pc = PC(ci_estimator=self.ci_estimator, apply_orientations=False)
        pc.fit(sample)
        skel_graph = pc.graph_
        print(skel_graph)
        assert list(skel_graph.undirected_edges) == [("x", "y"), ("z", "y")]

    def test_pc_basic_collider(self):
        sample = self.scm.sample(n=1, include_latents=False)
        pc = PC(ci_estimator=self.ci_estimator, apply_orientations=False)
        pc.fit(sample)
        skel_graph = pc.graph_
        sep_set = pc.separating_sets_
        self.alg._orient_colliders(skel_graph, sep_set)

        # the CPDAG learned should be a fully oriented DAG
        assert len(skel_graph.undirected_edges) == 0
        assert skel_graph.has_edge("x", "y")
        assert skel_graph.has_edge("z", "y")

    def test_pc_rule1(self):
        # If C -> A - B, such that the triple is unshielded,
        # then orient A - B as A -> B

        G = CPDAG()
        G.add_undirected_edge("A", "B")
        G.add_edge("C", "A")
        G_copy = G.copy()

        # First test:
        # C -> A -> B
        self.alg._apply_rule1(G, "A", "B")
        assert G.has_edge("A", "B")
        assert not G.has_undirected_edge("A", "B")
        assert G.has_edge("C", "A")

        # Next, test that nothing occurs
        # if it is a shielded triple
        G = G_copy.copy()
        G.add_edge("C", "B")
        added_arrows = self.alg._apply_rule1(G, "A", "B")
        assert not added_arrows

    def test_pc_rule2(self):
        # If C -> A -> B, with C - B
        # then orient C - B as C -> B

        G = CPDAG()
        G.add_undirected_edge("C", "B")
        G.add_edge("C", "A")
        G.add_edge("A", "B")

        # First test:
        # C -> B
        added_arrows = self.alg._apply_rule2(G, "C", "B")
        assert G.has_edge("A", "B")
        assert G.has_edge("C", "A")
        assert G.has_edge("C", "B")
        assert added_arrows

    def test_pc_rule3(self):
        # If C - A, with C - B and C - D
        # and B -> A <- D, then orient C - A as C -> A.

        G = CPDAG()
        G.add_undirected_edge("C", "B")
        G.add_undirected_edge("C", "D")
        G.add_undirected_edge("C", "A")
        G.add_edge("B", "A")
        G.add_edge("D", "A")
        G_copy = G.copy()

        added_arrows = self.alg._apply_rule3(G, "C", "A")
        assert added_arrows
        assert G.has_edge("C", "A")

        # if 'B' and 'D' are adjacent, then it will
        # not be added
        G = G_copy.copy()
        G.add_edge("B", "D")
        added_arrows = self.alg._apply_rule3(G, "C", "A")
        assert not added_arrows
