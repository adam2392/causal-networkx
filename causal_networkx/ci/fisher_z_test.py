from math import log, sqrt
from typing import Any, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from .base import BaseConditionalIndependenceTest


class FisherZCITest(BaseConditionalIndependenceTest):
    def __init__(self, correlation_matrix=None):
        """Conditional independence test using Fisher-Z's test for Gaussian random variables.

        Parameters
        ----------
        correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
            ``None`` means without the parameter of correlation matrix and
            the correlation will be computed from the data., by default None
        """
        self.correlation_matrix = correlation_matrix

    def test(self, df: pd.DataFrame, x: Any, y: Any, z: Any = None) -> Tuple[float, float]:
        """Run conditional independence test.

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        x : Any
            _description_
        y : Any
            _description_
        z : Any, optional
            _description_, by default None

        Returns
        -------
        stat : float
            The test statistic.
        pvalue : float
            The p-value of the test.
        """
        if z is None:
            z = set()
        stat, pvalue = fisherz(df, x, y, z, self.correlation_matrix)
        return stat, pvalue


def fisherz(
    data: Union[NDArray, pd.DataFrame],
    x: Union[int, str],
    y: Union[int, str],
    sep_set: Set,
    correlation_matrix=None,
):
    """Perform an independence test using Fisher-Z's test.

    Works on Gaussian random variables.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    x : int | str
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : int | str
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).
    correlation_matrix : np.ndarray of shape (n_variables, n_variables), optional
            ``None`` means without the parameter of correlation matrix and
            the correlation will be computed from the data., by default None

    Returns
    -------
    X : float
        The test statistic.
    p : float
        The p-value of the test.
    """
    data_arr = data.to_numpy()

    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data_arr.T)
    sample_size = data.shape[0]
    var = list({x, y}.union(sep_set))  # type: ignore
    (var_idx,) = np.in1d(data.columns, var).nonzero()

    # compute the correlation matrix within the specified data
    sub_corr_matrix = correlation_matrix[np.ix_(var_idx, var_idx)]
    inv = np.linalg.inv(sub_corr_matrix)
    r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
    Z = 0.5 * log((1 + r) / (1 - r))
    X = sqrt(sample_size - len(sep_set) - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(X)))
    return X, p
