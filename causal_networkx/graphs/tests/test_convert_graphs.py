import pytest

from causal_networkx import DAG


def test_convert_latent_to_unobserved_confounders():
    # build dict-of-dict-of-dict for 1 <- 0 -> 2
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}}
    G = DAG(incoming_graph_data)

    with pytest.raises(RuntimeError, match="not a common cause"):
        G.set_nodes_as_latent_confounders([1])

    # setting nodes as latent confounders will remove that node
    # and form a new set of bidirected edges
    admg = G.set_nodes_as_latent_confounders([0])
    assert set(admg.nodes) == {1, 2}
    assert set(admg.bidirected_edges) == {(1, 2)}
    assert set(admg.edges) == set()

    # adding a parent to the new latent confounder, shifts that
    # edge to directly affect the confounder's children
    G.add_edge(3, 0)
    admg = G.set_nodes_as_latent_confounders([0])
    assert set(admg.nodes) == {1, 2, 3}
    assert set(admg.bidirected_edges) == {(1, 2)}
    assert set(admg.edges) == {(3, 1), (3, 2)}
