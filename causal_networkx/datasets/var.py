from collections import defaultdict


def _iter_coeffs(parents_neighbors_coeffs):
    """
    Iterator through the current parents_neighbors_coeffs structure.  Mainly to
    save repeated code and make it easier to change this structure.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Yields
    -------
    (node_id, parent_id, time_lag, coeff) : tuple
        Tuple defining the relationship between nodes across time
    """
    # Iterate through all defined nodes
    for node_id in list(parents_neighbors_coeffs):
        # Iterate over parent nodes and unpack node and coeff
        for (parent_id, time_lag), coeff in parents_neighbors_coeffs[node_id]:
            # Yield the entry
            yield node_id, parent_id, time_lag, coeff


def _check_parent_neighbor(parents_neighbors_coeffs):
    """
    Checks to insure input parent-neighbor connectivity input is sane.  This
    means that:
        * all time lags are non-positive
        * all parent nodes are included as nodes themselves
        * all node indexing is contiguous
        * all node indexing starts from zero
    Raises a ValueError if any one of these conditions are not met.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.
    """
    # Initialize some lists for checking later
    all_nodes = set()
    all_parents = set()
    # Iterate through variables
    for j in list(parents_neighbors_coeffs):
        # Cache all node ids to ensure they are contiguous
        all_nodes.add(j)
    # Iterate through all nodes
    for j, i, tau, _ in _iter_coeffs(parents_neighbors_coeffs):
        # Check all time lags are equal to or less than zero
        if tau > 0:
            raise ValueError(
                "Lag between parent {} and node {}".format(i, j)
                + " is {} > 0, must be <= 0!".format(tau)
            )
        # Cache all parent ids to ensure they are mentioned as node ids
        all_parents.add(i)
    # Check that all nodes are contiguous from zero
    all_nodes_list = sorted(list(all_nodes))
    if all_nodes_list != list(range(len(all_nodes_list))):
        raise ValueError(
            "Node IDs in input dictionary must be contiguous"
            + " and start from zero!\n"
            + " Found IDs : ["
            + ",".join(map(str, all_nodes_list))
            + "]"
        )
    # Check that all parent nodes are mentioned as a node ID
    if not all_parents.issubset(all_nodes):
        missing_nodes = sorted(list(all_parents - all_nodes))
        all_parents_list = sorted(list(all_parents))
        raise ValueError(
            "Parent IDs in input dictionary must also be in set"
            + " of node IDs."
            + "\n Parent IDs "
            + " ".join(map(str, all_parents_list))
            + "\n Node IDs "
            + " ".join(map(str, all_nodes_list))
            + "\n Missing IDs "
            + " ".join(map(str, missing_nodes))
        )


def _get_true_parent_neighbor_dict(parents_neighbors_coeffs):
    """
    Function to return the dictionary of true parent neighbor causal
    connections in time.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format:
        {..., j:[((var1, lag1), coef1), ((var2, lag2), coef2), ...], ...} for
        all variables where vars must be in [0..N-1] and lags <= 0 with number
        of variables N.

    Returns
    -------
    true_parent_neighbor : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Initialize the returned dictionary of lists
    true_parents_neighbors = defaultdict(list)
    for j in parents_neighbors_coeffs:
        for link_props in parents_neighbors_coeffs[j]:
            i, tau = link_props[0]
            coeff = link_props[1]
            # Add parent node id and lag if non-zero coeff
            if coeff != 0.0:
                true_parents_neighbors[j].append((i, tau))
    # Return the true relations
    return true_parents_neighbors


def var_process(
    parents_neighbors_coeffs, T=1000, use="inv_inno_cov", verbosity=0, initial_values=None
):
    """Returns a vector-autoregressive process with correlated innovations.

    Wrapper around var_network with possibly more user-friendly input options.

    Parameters
    ----------
    parents_neighbors_coeffs : dict
        Dictionary of format: {..., j:[((var1, lag1), coef1), ((var2, lag2),
        coef2), ...], ...} for all variables where vars must be in [0..N-1]
        and lags <= 0 with number of variables N. If lag=0, a nonzero value
        in the covariance matrix (or its inverse) is implied. These should be
        the same for (i, j) and (j, i).
    use : str, optional (default: 'inv_inno_cov')
        Specifier, either 'inno_cov' or 'inv_inno_cov'.
        Any other specifier will result in non-correlated noise.
        For debugging, 'no_noise' can also be specified, in which case random
        noise will be disabled.
    T : int, optional (default: 1000)
        Sample size.
    verbosity : int, optional (default: 0)
        Level of verbosity.
    initial_values : array, optional (default: None)
        Initial values for each node. Shape must be (N, max_delay+1)

    Returns
    -------
    data : array-like
        Data generated from this process
    true_parent_neighbor : dict
        Dictionary of lists of tuples.  The dictionary is keyed by node ID, the
        list stores the tuple values (parent_node_id, time_lag)
    """
    # Check the input parents_neighbors_coeffs dictionary for sanity
    _check_parent_neighbor(parents_neighbors_coeffs)
    # Generate the true parent neighbors graph
    true_parents_neighbors = _get_true_parent_neighbor_dict(parents_neighbors_coeffs)
    # Generate the correlated innovations
    innos = _get_covariance_matrix(parents_neighbors_coeffs)
    # Generate the lagged connectivity matrix for _var_network
    connect_matrix = _get_lag_connect_matrix(parents_neighbors_coeffs)
    # Default values as per 'inno_cov'
    add_noise = True
    invert_inno = False
    # Use the correlated innovations
    if use == "inno_cov":
        if verbosity > 0:
            print("\nInnovation Cov =\n%s" % str(innos))
    # Use the inverted correlated innovations
    elif use == "inv_inno_cov":
        invert_inno = True
        if verbosity > 0:
            print("\nInverse Innovation Cov =\n%s" % str(innos))
    # Do not use any noise
    elif use == "no_noise":
        add_noise = False
        if verbosity > 0:
            print("\nInverse Innovation Cov =\n%s" % str(innos))
    # Use decorrelated noise
    else:
        innos = None
    # Ensure the innovation matrix is symmetric if it is used
    if (innos is not None) and add_noise:
        _check_symmetric_relations(innos)
    # Generate the data using _var_network
    data = _var_network(
        graph=connect_matrix,
        add_noise=add_noise,
        inno_cov=innos,
        invert_inno=invert_inno,
        T=T,
        initial_values=initial_values,
    )
    # Return the data
    return data, true_parents_neighbors
