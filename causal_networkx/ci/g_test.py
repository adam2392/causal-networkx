# This code was originally adapted from https://github.com/keiichishima/gsq
# and heavily refactored and modified.
from typing import List, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2


def _calculate_contingency_tble(
    x: Union[int, str],
    y: Union[int, str],
    sep_set: Union[List, Set],
    dof: int,
    data: NDArray,
    nlevels_x: int,
    nlevels_y: int,
    levels: NDArray = None,
) -> NDArray:
    """Calculate log term for binary G^2 statistic in CI test.

    Computes the contingency table and the associated log-term
    within the G^2 statistic for binary data.

    Parameters
    ----------
    x : int | str
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : int | str
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).
    dof : int
        The degrees of freedom.
    data : np.ndarray of shape (n_samples, n_variables)
        The input data matrix.
    nlevels_x : int
        The number of levels in the 'x' data variable.
    nlevels_y : int
        The number of levels in the 'y' data variable.
    levels : np.ndarray of shape (n_variables,)
        The number of levels associated with each variable.

    Returns
    -------
    contingency_tble : np.ndarray of shape (2, 2, dof)
        A contingency table per degree of freedom.
    """
    # define contingency table as a 2 by 2 table relating 'x' and 'y'
    # across different separating set variables
    contingency_tble = np.zeros((nlevels_x, nlevels_y, dof))
    x_idx = data[x]  # [:, x]
    y_idx = data[y]  # [:, y]
    sep_set = list(sep_set)

    # sum all co-occurrences of x and y conditioned on z
    for row_idx, (idx, jdx) in enumerate(zip(x_idx, y_idx)):
        kdx = 0
        for zidx, z in enumerate(sep_set):
            if levels is None:
                # binary case
                kdx += data[z][row_idx] * int(pow(2, zidx))
            else:
                # discrete case
                if zidx == 0:
                    kdx += data[z][row_idx]  # data[row_idx, z]
                else:
                    lprod = np.prod(list(map(lambda x: levels[x], sep_set[:zidx])))  # type: ignore
                    kdx += data[z][row_idx] * lprod

        # increment the co-occurrence found
        contingency_tble[idx, jdx, kdx] += 1
    return contingency_tble


def _calculate_highdim_contingency(
    x: Union[int, str],
    y: Union[int, str],
    sep_set: Set,
    data: NDArray,
    nlevel_x: int,
    nlevels_y: int,
) -> NDArray:
    """Calculate the contingency table for "large" separating set.

    When separating set is deemed "large", we use a different approach
    to computing the overall contingency table.

    Parameters
    ----------
    x : int | str
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : int | str
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).
    data : np.ndarray of shape (n_samples, n_variables)
        The input data matrix.
    nlevel_x : int
        Number of levels of the 'x' variable in the data matrix.
    nlevels_y : int
        Number of levels of the 'y' variable in the data matrix.

    Returns
    -------
    contingency_tble : np.ndarray of shape (nlevel_x, nlevel_y, dof)
        A contingency table per degree of freedom per level
        of each variable.
    """
    n_samples, _ = data.shape

    # keep track of all variables in the separating set
    sep_set = list(sep_set)
    k = data[:, sep_set]

    # count number of value combinations for sepset variables
    # observed in the data
    dof_count = 1
    parents_val = np.array([k[0, :]])

    # initialize the contingency table
    contingency_tble = np.zeros((2, 2, 1))
    xdx = data[0, x]
    ydx = data[0, y]
    contingency_tble[xdx, ydx, dof_count - 1] = 1

    # check how many parents we can create from the rest of the dataset
    for idx in range(1, n_samples):
        is_new = True
        xdx = data[idx, x]
        ydx = data[idx, y]

        # comparing the current values of the subset variables to all
        # already existing combinations of subset variables values
        tcomp = parents_val[:dof_count, :] == k[idx, :]
        for it_parents in range(dof_count):
            if np.all(tcomp[it_parents, :]):
                contingency_tble[xdx, ydx, it_parents] += 1
                is_new = False
                break

        # new combination of separating set values, so we create a new
        # contingency table
        if is_new:
            dof_count += 1
            parents_val = np.r_[parents_val, [k[idx, :]]]

            # create a new contingnecy table and update cell counts
            # using the original table up to the last value
            ncontingency_tble = np.zeros((nlevel_x, nlevels_y, dof_count))
            for p in range(dof_count - 1):
                ncontingency_tble[:, :, p] = contingency_tble[:, :, p]
            ncontingency_tble[xdx, ydx, dof_count - 1] = 1
            contingency_tble = ncontingency_tble
    return contingency_tble


def _calculate_g_statistic(contingency_tble):
    """Calculate a G statistic from contingency table.

    Parameters
    ----------
    contingency_tble : np.ndarray of shape (nlevels_x, nlevels_y, dof)
        The contingency table of 'x' vs 'y'.

    Returns
    -------
    G2 : float
        G^2 test statistic.
    """
    nlevels_x, nlevels_y, dof_count = contingency_tble.shape

    # now compute marginal terms across all degrees of freedom
    tx_dof = contingency_tble.sum(axis=1)
    ty_dof = contingency_tble.sum(axis=0)

    # compute sample size within each degree of freedom
    nk = ty_dof.sum(axis=0)

    # compute the term to be logged:
    # s^{ab}_{ij} * M / (s_i^a s_j^b)
    tlog = np.zeros((nlevels_x, nlevels_y, dof_count))
    for k in range(dof_count):
        # create a 2x1 and 1x2 array of marginal counts
        # for each degree of freedom
        tx = tx_dof[..., k][:, np.newaxis]
        ty = ty_dof[..., k][np.newaxis, :]

        # compute the final term in the log
        tdijk = tx.dot(ty)
        tlog[:, :, k] = contingency_tble[:, :, k] * nk[k] / tdijk

    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * contingency_tble * log_tlog)
    return G2


def g_square_binary(
    data: Union[NDArray, pd.DataFrame], x: Union[int, str], y: Union[int, str], sep_set: Set
) -> Tuple[float, float]:
    """G square test for a binary data.

    When running a conditional-independence test, degrees of freecom
    is calculated. It is defined as 2^|S|, where |S| is the
    cardinality of the separating set, S.

    Parameters
    ----------
    data : np.ndarray | pandas.DataFrame of shape (n_samples, n_variables)
        The data matrix to be used.
    x : int | str
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : int | str
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).

    Returns
    -------
    G2_stat : float
        The G^2 statistic.
    p_val : float
        The p-value of conditional independence.

    Notes
    -----
    The G^2 statistic for binary outcome 'a' and 'b' is:

    ..math::

        2 * \sum_{a,b} S^{a,b}_{ij} ln(\frac{s^{ab}_{ij} M}{s_i^a s_j^b})

    which takes the sum over occurrences of 'a' and 'b' and multiplies
    it by the number of samples, M and normalizes it.

    References
    ----------
    See: http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    if any(xy not in data.columns for xy in [x, y]):
        raise ValueError(
            f'Variables "x" ({x}) and "y" ({y}) are not in the columns of "data": {data.columns.values}.'
        )

    n_samples = data.shape[0]
    s_size = len(sep_set)
    dof = int(pow(2, s_size))

    # check number of samples relative to degrees of freedom
    # assuming no zeros
    n_samples_req = 10 * dof
    if n_samples < n_samples_req:
        raise RuntimeError(f"Not enough samples. {n_samples} is too small. Need {n_samples_req}.")

    # hard-cut off cardinality of separating set at 6
    if s_size < 6:
        # set up contingency table for binary data 2x2xdof
        contingency_tble = _calculate_contingency_tble(x, y, sep_set, dof, data, 2, 2)
    else:
        # s_size >= 6
        contingency_tble = _calculate_highdim_contingency(x, y, sep_set, data, 2, 2)

    G2_stat = _calculate_g_statistic(contingency_tble)

    p_val = chi2.sf(G2_stat, dof)
    return G2_stat, p_val


def g_square_discrete(
    data: Union[NDArray, pd.DataFrame],
    x: Union[int, str],
    y: Union[int, str],
    sep_set: Set,
    levels=None,
) -> Tuple[float, float]:
    """G square test for discrete data.

    Parameters
    ----------
    data : np.ndarray | pandas.DataFrame of shape (n_samples, n_variables)
        The data matrix to be used.
    x : int | str
        the first node variable. If ``data`` is a DataFrame, then
        'x' must be in the columns of ``data``.
    y : int | str
        the second node variable. If ``data`` is a DataFrame, then
        'y' must be in the columns of ``data``.
    sep_set : set
        the set of neibouring nodes of x and y (as a set()).
    levels: levels of each column in the data matrix
            (as a list()).

    Returns
    -------
    G2 : float
        The G^2 test statistic.
    p_val : float
        the p-value of conditional independence.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    if any(xy not in data.columns for xy in [x, y]):
        raise ValueError(
            f'Variables "x" ({x}) and "y" ({y}) are not in the columns of "data": {data.columns.values}.'
        )

    if levels is None:
        levels = np.amax(data, axis=0) + 1
    n_samples = data.shape[0]
    s_size = len(sep_set)
    dof = (levels[x] - 1) * (levels[y] - 1) * np.prod(list(map(lambda x: levels[x], sep_set)))

    # check number of samples relative to degrees of freedom
    n_samples_req = 10 * dof
    if n_samples < n_samples_req:
        raise RuntimeError(f"Not enough samples. {n_samples} is too small. Need {n_samples_req}.")

    contingency_tble = None
    # hard-cut off cardinality of separating set at 5
    if s_size < 5:
        # degrees of freedom
        prod_levels = np.prod(list(map(lambda x: levels[x], sep_set))).astype(int)

        # set up contingency table for binary data
        # |X| x |Y| x dof
        contingency_tble = _calculate_contingency_tble(
            x,
            y,
            sep_set,
            prod_levels,
            data,
            nlevels_x=levels[x],
            nlevels_y=levels[y],
            levels=levels,
        )
    else:
        # s_size >= 5
        contingency_tble = _calculate_highdim_contingency(x, y, sep_set, data, levels[x], levels[y])

    # calculate the actual G statistic given the contingency table
    G2 = _calculate_g_statistic(contingency_tble)

    if dof == 0:
        # dof can be 0 when levels[x] or levels[y] is 1, which is
        # the case that the values of columns x or y are all 0.
        p_val = 1
    else:
        p_val = chi2.sf(G2, dof)
    return G2, p_val
