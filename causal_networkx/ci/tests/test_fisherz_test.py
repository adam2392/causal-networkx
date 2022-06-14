import numpy as np
import pandas as pd

from causal_networkx.ci import FisherZCITest

seed = 12345


def test_fisher_z():
    """Test Fisher Z test for Gaussian data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    X1 = rng.randn(300, 1)
    Y = X + X1 + 0.5 * rng.randn(300, 1)
    Z = Y + 0.1 * rng.randn(300, 1)

    ci_estimator = FisherZCITest()

    # create input for the CI test
    df = pd.DataFrame(np.hstack((X, X1, Y, Z)), columns=["x", "x1", "y", "z"])

    _, pvalue = ci_estimator.test(df, "x", "x1")
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(df, "x", "z")
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(df, "x", "z", "y")
    assert pvalue > 0.05
