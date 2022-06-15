import networkx as nx

from causal_networkx import ADMG, DAG
from causal_networkx.algorithms import admg2pag, dag2cpdag


def test_dag2cpdag():
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = DAG(incoming_graph_data)

    cpdag = dag2cpdag(G)

    # the CPDAG should have the unshielded collider oriented
    # it should also have the same adjacency structure
    assert set(cpdag.edges) == {(3, 2), (0, 2)}
    assert nx.is_isomorphic(G.to_adjacency_graph(), cpdag.to_adjacency_graph())


def test_admg2pag():
    # build initial DAG
    ed1, ed2 = ({}, {})
    incoming_graph_data = {0: {1: ed1, 2: ed2}, 3: {2: ed2}}
    G = ADMG(incoming_graph_data)

    # remove 0 and set a bidirected edge between 1 <--> 2
    # 1 <--> 2 <- 3, so 3 is independent of 1, but everything else is connected
    # the collider should be orientable.
    G = G.set_nodes_as_latent_confounders([0])
    pag = admg2pag(G)

    # the PAG should have the unshielded collider oriented
    # it should also have the same adjacency structure
    assert set(pag.edges) == {(3, 2), (1, 2)}
    assert nx.is_isomorphic(G.to_adjacency_graph(), pag.to_adjacency_graph())
