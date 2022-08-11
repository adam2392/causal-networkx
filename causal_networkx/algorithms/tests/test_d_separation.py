from causal_networkx import DAG
from causal_networkx.algorithms import (
    compute_minimal_separating_set,
    is_separating_set_minimal,
)
from causal_networkx.algorithms.d_separation import d_separated


def test_minimal_sep_set_parents():
    # Case 1:
    # create a graph A -> B <- C
    # B -> D -> E;
    # B -> F;
    # G -> E;
    edge_list = [("A", "B"), ("C", "B"), ("B", "D"), ("D", "E"), ("B", "F"), ("G", "E")]
    G = DAG(edge_list)

    assert not d_separated(G, "B", "E")

    # minimal set of the corresponding graph
    # for B and E should be (D,)
    Zmin = compute_minimal_separating_set(G, "B", "E")

    # the minimal separating set should pass the test for minimality
    assert is_separating_set_minimal(G, "B", "E", Zmin)
    assert Zmin == {"D"}

    # Case 2:
    # create a graph A -> B -> C
    # B -> D -> C;
    edge_list = [("A", "B"), ("B", "C"), ("B", "D"), ("D", "C")]
    G = DAG(edge_list)
    assert not d_separated(G, "A", "C")

    Zmin = compute_minimal_separating_set(G, "A", "C")

    # the minimal separating set should pass the test for minimality
    assert is_separating_set_minimal(G, "A", "C", Zmin)
    assert Zmin == {"B"}
