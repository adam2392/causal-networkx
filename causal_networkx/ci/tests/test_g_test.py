from math import frexp

import numpy as np
import pytest

from causal_networkx.ci import GSquareCITest
from causal_networkx.ci.tests import testdata


def test_g_error():
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]

    with pytest.raises(ValueError, match="data_type"):
        ci_estimator = GSquareCITest(data_type="auto")
        ci_estimator.test(dm, x, y, set(sets[0]), [3, 2, 3, 4, 2])


def test_g_discrete():
    """Test G^2 test for discrete data."""
    dm = np.array([testdata.dis_data]).reshape((10000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="discrete")

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(dm, x, y, set(sets[idx]), [3, 2, 3, 4, 2])
        fr_p = frexp(p)
        fr_a = frexp(testdata.dis_answer[idx])
        assert round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]

    # check error message for number of samples
    dm = np.array([testdata.dis_data]).reshape((2000, 25))
    levels = np.ones((25,)) * 3
    sets = [[2, 3, 4, 5, 6, 7]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(dm, x, y, set(sets[0]), levels)

    # check error message for input
    with pytest.raises(ValueError, match='Variables "x"'):
        ci_estimator.test(dm, "test", y, set(sets[0]), levels)


def test_g_binary():
    """Test G^2 test for binary data."""
    dm = np.array([testdata.bin_data]).reshape((5000, 5))
    x = 0
    y = 1
    ci_estimator = GSquareCITest(data_type="binary")

    sets = [[], [2], [2, 3], [3, 4], [2, 3, 4]]
    for idx in range(len(sets)):
        _, p = ci_estimator.test(dm, x, y, set(sets[idx]))
        fr_p = frexp(p)
        fr_a = frexp(testdata.bin_answer[idx])
        assert round(fr_p[0] - fr_a[0], 7) == 0 and fr_p[1] == fr_a[1]

    # check error message for number of samples
    dm = np.array([testdata.bin_data]).reshape((500, 50))
    sets = [[2, 3, 4, 5, 6, 7, 8]]
    with pytest.raises(RuntimeError, match="Not enough samples"):
        ci_estimator.test(dm, x, y, set(sets[0]))

    # check error message for input
    with pytest.raises(ValueError, match='Variables "x"'):
        ci_estimator.test(dm, "test", y, set(sets[0]))