from math import log, sqrt
from typing import Set, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm


def fisherz(
    data: Union[NDArray, pd.DataFrame],
    x: Union[int, str],
    y: Union[int, str],
    sep_set: Set,
    correlation_matrix=None,
):
    """Perform an independence test using Fisher-Z's test.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_variables)
        The data.
    x : int | str
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : int | str
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).
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
    var = list((x, y).union(sep_set))  # type: ignore
    sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(sample_size - len(sep_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    return p
