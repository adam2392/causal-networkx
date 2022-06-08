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
    # get all true edges
    # y_true = true_graph.edges
    # y_pred = pred_graph.edges

    # compute a score
    # scores =
    pass


def structure_hamming_dist(graph, other_graph):
    """Compute structural hamming distance.

    Parameters
    ----------
    graph : _type_
        _description_
    other_graph : _type_
        _description_
    """
    pass