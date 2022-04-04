from math import log, sqrt

import numpy as np
from scipy.stats import norm


def fisherz(data, X, Y, condition_set, correlation_matrix=None):
    """Perform an independence test using Fisher-Z's test.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_variables)
        The data.
    X, Y and condition_set : column indices of data
    correlation_matrix : np.ndarray of shape (n_variables, n_variables
        ``None`` means without the parameter of correlation matrix and
        the correlation will be computed from the data.

    Returns
    -------
    p : the p-value of the test
    """
    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data.T)
    sample_size = data.shape[0]
    var = list((X, Y) + condition_set)
    sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(sample_size - len(condition_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    return p
