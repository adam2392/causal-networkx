import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from causal_networkx.graphs.base import BaseGraph


def confusion_matrix_networks(
    true_graph,
    pred_graph,
):
    """Compute the confusion matrix comparing a predicted graph from the true graph.

    Parameters
    ----------
    true_graph : an instance of causal graph
        The true graph.
    pred_graph : an instance of causal graph
        The predicted graph. The predicted graph and true graph must be
        the same type.

    Returns
    -------
    cm : np.ndarray of shape (2, 2)
        The confusion matrix.
    """
    assert list(true_graph.nodes) == list(pred_graph.nodes)

    # convert graphs to adjacency graph in networkx
    if isinstance(true_graph, BaseGraph):
        true_graph = true_graph.to_adjacency_graph()
    if isinstance(pred_graph, BaseGraph):
        pred_graph = pred_graph.to_adjacency_graph()

    # next convert into 2D numpy array format
    true_adj_mat = nx.to_numpy_array(true_graph)
    pred_adj_mat = nx.to_numpy_array(pred_graph)

    # ensure we are looking at symmetric graphs
    true_adj_mat += true_adj_mat.T
    pred_adj_mat += pred_adj_mat.T

    # then only extract lower-triangular portion
    true_adj_mat = true_adj_mat[np.tril_indices_from(true_adj_mat, k=-1)]
    pred_adj_mat = pred_adj_mat[np.tril_indices_from(pred_adj_mat, k=-1)]

    true_adj_mat = true_adj_mat > 0
    pred_adj_mat = pred_adj_mat > 0

    # vectorize and binarize for sklearn's confusion matrix
    y_true = LabelBinarizer().fit_transform(true_adj_mat.flatten()).squeeze()
    y_pred = LabelBinarizer().fit_transform(pred_adj_mat.flatten()).squeeze()

    # compute the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    return conf_mat


def structure_hamming_dist(graph, other_graph, double_for_anticausal: bool = True):
    """Compute structural hamming distance.

    Parameters
    ----------
    graph : _type_
        Reference graph.
    other_graph : _type_
        Other graph.
    double_for_anticausal : bool, optional
        Whether to count incorrect orientations as two mistakes, by default True

    Returns
    -------
    shm : float
        The hamming distance between 0 and infinity.
    """
    if isinstance(graph, BaseGraph):
        graph = graph.to_networkx()  # type: ignore
    if isinstance(other_graph, BaseGraph):
        other_graph = other_graph.to_networkx()  # type: ignore
    # convert graphs to adjacency matrix in numpy array format
    adj_mat = nx.to_numpy_array(graph)
    other_adj_mat = nx.to_numpy_array(other_graph)

    diff = np.abs(adj_mat - other_adj_mat)

    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2
