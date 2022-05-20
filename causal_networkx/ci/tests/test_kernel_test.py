import numpy as np
import pytest

from causal_networkx.ci import KernelCITest

seed = 12345
ci_params = {
    "rbf_approx": KernelCITest(),
    "rbf": KernelCITest(approx_with_gamma=False),
    "linear_approx": KernelCITest(
        kernel_x="linear",
        kernel_y="linear",
        kernel_z="linear",
    ),
    "linear": KernelCITest(
        kernel_x="linear", kernel_y="linear", kernel_z="linear", approx_with_gamma=False
    ),
    "polynomial_approx": KernelCITest(
        kernel_x="polynomial",
        kernel_y="polynomial",
        kernel_z="polynomial",
    ),
    "polynomial": KernelCITest(
        kernel_x="polynomial", kernel_y="polynomial", kernel_z="polynomial", approx_with_gamma=False
    ),
}


@pytest.mark.parametrize("ci_estimator", ci_params.values(), ids=ci_params.keys())
def test_kci_with_gaussian_data(ci_estimator):
    rng = np.random.RandomState(seed)
    X = rng.randn(300, 1)
    X1 = rng.randn(300, 1)
    Y = np.concatenate((X, X), axis=1) + 0.5 * rng.randn(300, 2)
    Z = Y + 0.5 * rng.randn(300, 2)

    _, pvalue = ci_estimator.test(X, X1)
    assert pvalue > 0.05
    _, pvalue = ci_estimator.test(X, Z)
    assert pvalue < 0.05
    _, pvalue = ci_estimator.test(X, Z, Y)
    assert pvalue > 0.05


def test_kci_errors():
    with pytest.raises(ValueError, match="The kernels that are currently supported"):
        KernelCITest(kernel_x="gauss")
