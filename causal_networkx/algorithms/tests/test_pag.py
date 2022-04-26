from itertools import permutations

from causal_networkx.algorithms import (
    discriminating_path,
    possibly_d_sep_sets,
    uncovered_pd_path,
)
from causal_networkx.cgm import PAG, CausalGraph
from causal_networkx.ci import Oracle
from causal_networkx.discovery import FCI


def test_possibly_d_separated():
    """Test possibly d-separated set construction.

    Uses Figure 15, 16, 17 and 18 in "Discovery Algorithms without
    Causal Sufficiency" in [1].

    References
    ----------
    [1] Spirtes, P., Glymour, C. and Scheines, R. (2000). Causation,
    Prediction, and Search, 2nd ed. MIT Press, Cambridge, MA. MR1815675
    """
    edge_list = [
        ("D", "A"),
        ("B", "E"),
        ("H", "D"),
        ("F", "B"),
    ]
    latent_edge_list = [("A", "B"), ("D", "E")]
    uncertain_edge_list = [
        ("A", "E"),
        ("E", "A"),
        ("E", "B"),
        ("B", "F"),
        ("F", "C"),
        ("C", "F"),
        ("C", "H"),
        ("H", "C"),
        ("D", "H"),
        ("A", "D"),
    ]
    G = PAG(edge_list, latent_edge_list, uncertain_edge_list)

    a_pdsep = possibly_d_sep_sets(G, "A", "E")
    e_pdsep = possibly_d_sep_sets(G, "E", "A")

    # reconstruct the PAG the way FCI would
    edge_list = [("D", "A"), ("B", "E"), ("F", "B"), ("C", "F"), ("C", "H"), ("H", "D")]
    latent_edge_list = [("A", "B"), ("D", "E")]
    graph = CausalGraph(edge_list, latent_edge_list)
    alg = FCI(ci_estimator=Oracle(graph).ci_test)
    sample = graph.dummy_sample()
    skel_graph, sep_set = alg.learn_skeleton(sample)
    fci_pag = PAG(incoming_uncertain_data=skel_graph)
    alg._orient_colliders(fci_pag, sep_set)

    # possibly d-sep sets should match
    pdsep = possibly_d_sep_sets(fci_pag, "A", "E")
    assert pdsep == a_pdsep
    pdsep = possibly_d_sep_sets(fci_pag, "E", "A")
    assert pdsep == e_pdsep


def test_discriminating_path():
    """Test the output of a discriminating path.

    We look at a graph presented in [1] Figure 2.

    References
    ----------
    [1] Colombo, Diego, et al. "Learning high-dimensional directed acyclic
    graphs with latent and selection variables." The Annals of Statistics
    (2012): 294-321.
    """
    # this is Figure 2's PAG after orienting colliders, there should be no
    # discriminating path
    edges = [
        ("x4", "x1"),
        ("x4", "x6"),
        ("x2", "x5"),
        ("x2", "x6"),
        ("x5", "x6"),
        ("x3", "x4"),
        ("x3", "x2"),
        ("x3", "x6"),
    ]
    bidirected_edges = [("x1", "x2"), ("x4", "x5")]
    circle_edges = [("x4", "x3"), ("x2", "x3"), ("x6", "x2"), ("x6", "x5"), ("x6", "x4")]
    pag = PAG(edges, bidirected_edges, circle_edges)

    for u in pag.nodes:
        for (a, c) in permutations(pag.neighbors(u), 2):
            _, found_discriminating_path, disc_path = discriminating_path(
                pag, u, a, c, max_path_length=100
            )
            if (c, u, a) == ("x6", "x3", "x2"):
                assert found_discriminating_path
            else:
                assert not found_discriminating_path

    # by making x5 <- x2 into x5 <-> x2, we will have another discriminating path
    pag.remove_edge("x2", "x5")
    pag.add_bidirected_edge("x5", "x2")
    for u in pag.nodes:
        for (a, c) in permutations(pag.neighbors(u), 2):
            _, found_discriminating_path, disc_path = discriminating_path(
                pag, u, a, c, max_path_length=100
            )
            if (c, u, a) in (("x6", "x5", "x2"), ("x6", "x3", "x2")):
                assert found_discriminating_path
            else:
                assert not found_discriminating_path

    edges = [
        ("x4", "x1"),
        ("x4", "x6"),
        ("x2", "x5"),
        ("x2", "x6"),
        ("x5", "x6"),
        ("x3", "x4"),
        ("x3", "x2"),
        ("x3", "x6"),
    ]
    bidirected_edges = [("x1", "x2"), ("x4", "x5")]
    circle_edges = [("x4", "x3"), ("x2", "x3"), ("x6", "x4"), ("x6", "x5"), ("x6", "x3")]
    pag = PAG(edges, bidirected_edges, circle_edges)
    _, found_discriminating_path, _ = discriminating_path(
        pag, "x3", "x2", "x6", max_path_length=100
    )
    assert found_discriminating_path


def test_uncovered_pd_path():
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

    # get the uncovered pd paths
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", 100, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]

    # the shielded triple should not result in an uncovered pd path
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "v", "C", 100, "A")
    assert not found_uncovered_pd_path
    assert uncov_pd_path == []

    # when there is a circle edge it should still work
    G.add_circle_edge("C", "z")
    uncov_pd_path, found_uncovered_pd_path = uncovered_pd_path(G, "u", "C", 100, "A")
    assert found_uncovered_pd_path
    assert uncov_pd_path == ["A", "u", "x", "y", "z", "C"]
