import numpy as np
from numpy import sqrt
from numpy.linalg import eigh, eigvalsh
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel


class KernelCITest:
    def __init__(
        self,
        kernel_x="gaussian",
        kernel_y="gaussian",
        kernel_z="gaussian",
        null_size=1000,
        approx_with_gamma=True,
        est_width="empirical",
        poly_dof=1,
        kwidth_x=None,
        kwidth_y=None,
        kwidth_z=None,
    ) -> None:
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.kernel_z = kernel_z
        self.null_size = null_size
        self.approx_with_gamma = approx_with_gamma
        self.est_width = est_width
        self.poly_dof = poly_dof
        self.kwidth_x = kwidth_x
        self.kwidth_y = kwidth_y
        self.kwidth_z = kwidth_z

    def test(self, X, Y, Z, n_reps=1000):
        # compute the kernel matrix of the data
        Kx, Kzx = self._compute_kernel_matrix(
            X, Z, self.kernel_x, self.kernel_z, self.kwidth_x, self.kwidth_z
        )

    def _compute_kernel_matrix(
        self, data, conditional_data, kernel, conditional_kernel, kwidth_data, kwdith_cond
    ):
        # first normalize the data to have zero mean and unit variance
        # along the columns of the data
        data = stats.zscore(data, axis=0)
        conditional_data = stats.zscore(conditional_data, axis=0)
