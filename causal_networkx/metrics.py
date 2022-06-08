import numpy as np
from sklearn.metrics import confusion_matrix


def compare_networks(
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

    # convert graphs to adjacency matrix in numpy array format
    true_adj_mat = true_graph.to_adjacency_graph()
    pred_adj_mat = pred_graph.to_adjacency_graph()

    true_adj_mat = true_adj_mat > 0
    pred_adj_mat = pred_adj_mat > 0

    # vectorize
    y_true = true_adj_mat.flatten()
    y_pred = pred_adj_mat.flatten()

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
    # convert graphs to adjacency matrix in numpy array format
    adj_mat = graph.to_adjacency_graph()
    other_adj_mat = other_graph.to_adjacency_graph()

    diff = np.abs(adj_mat - other_adj_mat)

    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2
