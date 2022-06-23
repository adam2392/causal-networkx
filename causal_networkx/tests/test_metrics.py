import numpy as np
from numpy.testing import assert_array_equal

from causal_networkx import CPDAG, DAG
from causal_networkx.metrics import (
    confusion_matrix_networks,
    graph_to_pred_vector,
    structure_hamming_dist,
)


def test_graph_to_pred_vector():
    edges = [(0, 1)]
    nodes = [0, 1, 2]
    G = DAG(edges)
    G.add_nodes_from(nodes)

    # test DAG to prediction vector
    pred_vec = graph_to_pred_vector(G)
    assert_array_equal(pred_vec, np.array([1, 0, 0]))

    # test CDAG to prediction vector
    G = CPDAG(G.dag)
    pred_vec = graph_to_pred_vector(G)
    assert_array_equal(pred_vec, np.array([1, 0, 0]))

    # test CPDAG to prediction vector with an undirected edges
    G.add_undirected_edge(0, 2)
    pred_vec = graph_to_pred_vector(G)
    assert_array_equal(pred_vec, np.array([1, 1, 0]))

    # TODO: make it work
    # if we want to output to a prediction vector accounting for edge direction
    # pred_vec = graph_to_pred_vector(G)
    # assert_array_equal(pred_vec, np.array([1, 1, 0]))


def test_confusion_matrix_networks():
    edges = [(0, 1)]
    nodes = [0, 1, 2]
    G = CPDAG(edges)
    G.add_nodes_from(nodes)

    edges = [(0, 2)]
    test_G = CPDAG(edges)
    test_G.add_nodes_from(nodes)

    # compute the confusion matrix
    conf_mat = confusion_matrix_networks(G, test_G)
    expected_cm = np.array([[1, 1], [1, 0]])
    assert_array_equal(expected_cm, conf_mat)


def test_structure_hamming_dist():
    """Test structural hamming distance computation using graphs."""
    edges = [(0, 1)]
    nodes = [0, 1, 2]
    G = DAG(edges)
    G.add_nodes_from(nodes)

    edges = [(0, 2)]
    test_G = DAG(edges)
    test_G.add_nodes_from(nodes)

    # compare the two graphs, which have two differing edges
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=False)
    assert shd == 2.0

    # adding the edge should reduce the distance
    G.add_edge(0, 2)
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=False)
    assert shd == 1.0

    # anticausal direction shouldnt matter
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=True)
    assert shd == 1.0

    # adding an edge in the wrong direction though will add double distance
    G.remove_edge(0, 2)
    G.add_edge(2, 0)
    shd = structure_hamming_dist(G, test_G, double_for_anticausal=True)
    assert shd == 3.0
