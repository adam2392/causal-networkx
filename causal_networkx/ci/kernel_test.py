import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS

from .base import BaseConditionalIndependenceTest


class KernelCITest(BaseConditionalIndependenceTest):
    def __init__(
        self,
        kernel_x: str = "rbf",
        kernel_y: str = "rbf",
        kernel_z: str = "rbf",
        null_size: int = 1000,
        approx_with_gamma: bool = True,
        kwidth_x=None,
        kwidth_y=None,
        kwidth_z=None,
        threshold: float = 1e-5,
        n_jobs: int = None,
    ):
        """Kernel (Conditional) Independence Test.

        For testing (conditional) independence on continuous data, we
        leverage kernels :footcite:`Zhang2011` that are computationally efficient.

        Parameters
        ----------
        kernel_x : str, optional
            The kernel function for data 'X', by default "rbf".
        kernel_y : str, optional
            The kernel function for data 'Y', by default "rbf".
        kernel_z : str, optional
            The kernel function for data 'Z', by default "rbf".
        null_size : int, optional
            The number of samples to generate for the bootstrap distribution to
            approximate the pvalue, by default 1000.
        approx_with_gamma : bool, optional
            Whether to use the Gamma distribution approximation for the pvalue,
            by default True.
        kwidth_x : _type_, optional
            _description_, by default None
        kwidth_y : _type_, optional
            _description_, by default None
        kwidth_z : _type_, optional
            _description_, by default None
        threshold : float, optional
            The threshold set on the value of eigenvalues, by default 1e-5. Used
            to regularize the method.
        n_jobs : int, optional
            The number of CPUs to use, by default None.

        Notes
        -----
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,
            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        References
        ----------
        .. footbibliography::
        """
        if isinstance(kernel_x, str) and kernel_x not in PAIRWISE_KERNEL_FUNCTIONS:
            raise ValueError(
                f"The kernels that are currently supported are {PAIRWISE_KERNEL_FUNCTIONS}. "
                f"You passed in {kernel_x} for kernel_x."
            )
        if isinstance(kernel_y, str) and kernel_y not in PAIRWISE_KERNEL_FUNCTIONS:
            raise ValueError(
                f"The kernels that are currently supported are {PAIRWISE_KERNEL_FUNCTIONS}. "
                f"You passed in {kernel_y} for kernel_y."
            )
        if isinstance(kernel_z, str) and kernel_z not in PAIRWISE_KERNEL_FUNCTIONS:
            raise ValueError(
                f"The kernels that are currently supported are {PAIRWISE_KERNEL_FUNCTIONS}. "
                f"You passed in {kernel_z} for kernel_z."
            )
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.kernel_z = kernel_z
        self.null_size = null_size
        self.approx_with_gamma = approx_with_gamma

        self.threshold = threshold
        self.n_jobs = n_jobs

        # hyperparameters of the kernsl
        self.kwidth_x = kwidth_x
        self.kwidth_y = kwidth_y
        self.kwidth_z = kwidth_z

    def test(self, df, x_var, y_var, z_covariates=None):
        if any(col not in df.columns for col in [x_var, y_var]):
            raise ValueError("The x and y variables are not both in the DataFrame.")
        if z_covariates is not None and any(col not in df.columns for col in z_covariates):
            raise ValueError("The z conditioning set variables are not all in the DataFrame.")

        if z_covariates is None or len(z_covariates) == 0:
            Z = None
        else:
            z_covariates = list(z_covariates)
            Z = df[z_covariates].to_numpy().reshape((-1, len(z_covariates)))
        X = df[x_var].to_numpy()[:, np.newaxis]
        Y = df[y_var].to_numpy()[:, np.newaxis]

        # first normalize the data to have zero mean and unit variance
        # along the columns of the data
        X = stats.zscore(X, axis=0)
        Y = stats.zscore(Y, axis=0)
        if Z is not None:
            Z = stats.zscore(Z, axis=0)

            # when running CI, \ddot{X} comprises of (X, Z)
            X = np.concatenate((X, Z), axis=1)

            Kz, sigma_z = self._compute_kernel(
                Z,
                distance_metric="l2",
                metric=self.kernel_z,
                kwidth=self.kwidth_z,
                centered=True,
            )

        # compute the centralized kernel matrices of each the datasets
        Kx, sigma_x = self._compute_kernel(
            X,
            distance_metric="l2",
            metric=self.kernel_x,
            kwidth=self.kwidth_x,
            centered=True,
        )
        Ky, sigma_y = self._compute_kernel(
            Y, distance_metric="l2", metric=self.kernel_y, kwidth=self.kwidth_y, centered=True
        )

        if Z is None:
            # test statistic is just the normal bivariate independence
            # test statistic
            test_stat = self._compute_V_statistic(Kx, Ky)

            if self.approx_with_gamma:
                # approximate the pvalue using the Gamma distribution
                k_appr, theta_appr = self._approx_gamma_params_ind(Kx, Ky)
                pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
            else:
                null_samples = self._compute_null_ind(Kx, Ky, n_samples=self.null_size)
                pvalue = np.sum(null_samples > test_stat) / float(self.null_size)
        else:
            # compute the centralizing matrix for the kernels according to
            # conditioning set Z
            epsilon = 1e-7
            n = Kx.shape[0]
            Rz = epsilon * np.linalg.pinv(Kz + epsilon * np.eye(n))

            # compute the centralized kernel matrices
            KxzR = Rz.dot(Kx).dot(Rz)
            KyzR = Rz.dot(Ky).dot(Rz)

            # compute the conditional independence test statistic
            test_stat = self._compute_V_statistic(KxzR, KyzR)

            # compute the product of the eigenvectors
            uu_prod = self._compute_prod_eigvecs(KxzR, KyzR, threshold=self.threshold)

            if self.approx_with_gamma:
                # approximate the pvalue using the Gamma distribution
                k_appr, theta_appr = self._approx_gamma_params_ci(uu_prod)
                pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
            else:
                null_samples = self._compute_null_ci(uu_prod, self.null_size)
                pvalue = np.sum(null_samples > test_stat) / float(self.null_size)
        return test_stat, pvalue

    def _approx_gamma_params_ind(self, Kx, Ky):
        T = Kx.shape[0]
        mean_appr = np.trace(Kx) * np.trace(Ky) / T
        var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def _approx_gamma_params_ci(self, uu_prod):
        """Get parameters of the approximated Gamma distribution.

        Parameters
        ----------
        uu_prod : np.ndarray of shape (n_features, n_features)
            The product of the eigenvectors of Kx and Ky, the kernels
            on the input data, X and Y.

        Returns
        -------
        k_appr : float
            The shape parameter of the Gamma distribution.
        theta_appr : float
            The scale parameter of the Gamma distribution.

        Notes
        -----
        X ~ Gamma(k, theta) with a probability density function of the following:

        .. math::
            f(x; k, \\theta) = \\frac{x^{k-1} e^{-x / \\theta}}{\\theta^k \\Gamma(k)}

        where $\\Gamma(k)$ is the Gamma function evaluated at k. In this scenario
        k governs the shape of the pdf, while $\\theta$ governs more how spread out
        the data is.
        """
        # approximate the mean and the variance
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))

        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

    def _compute_prod_eigvecs(self, Kx, Ky, threshold=None):
        T = Kx.shape[0]
        wx, vx = np.linalg.eigh(0.5 * (Kx + Kx.T))
        wy, vy = np.linalg.eigh(0.5 * (Ky + Ky.T))

        if threshold is not None:
            # threshold eigenvalues that are below a certain threshold
            # and remove their corresponding values and eigenvectors
            vx = vx[:, wx > np.max(wx) * threshold]
            wx = wx[wx > np.max(wx) * threshold]
            vy = vy[:, wy > np.max(wy) * threshold]
            wy = wy[wy > np.max(wy) * threshold]

        # scale the eigenvectors by their eigenvalues
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # compute the product of the scaled eigenvectors
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((T, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                # compute the dot product of eigenvectors
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        # now compute the product
        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod

    def _compute_V_statistic(self, KxR, KyR):
        # n = KxR.shape[0]
        # compute the sum of the two kernsl
        Vstat = np.sum(KxR * KyR)
        return Vstat

    def _compute_null_ind(self, Kx, Ky, n_samples, max_num_eigs=1000):
        n = Kx.shape[0]

        # get the eigenvalues in ascending order, smallest to largest
        eigvals_x = np.linalg.eigvalsh(Kx)
        eigvals_y = np.linalg.eigvalsh(Ky)

        # optionally only keep the largest "N" eigenvalues
        eigvals_x = eigvals_x[-max_num_eigs:]
        eigvals_y = eigvals_y[-max_num_eigs:]
        num_eigs = len(eigvals_x)

        # compute the entry-wise product of the eigenvalues and store it as a vector
        eigvals_prod = np.dot(
            eigvals_x.reshape(num_eigs, 1), eigvals_y.reshape(1, num_eigs)
        ).reshape((-1, 1))

        # only keep eigenvalues above a certain threshold
        eigvals_prod = eigvals_prod[eigvals_prod > eigvals_prod.max() * self.threshold]

        # generate chi-square distributed values z_{ij} with degree of freedom 1
        f_rand = np.random.chisquare(df=1, size=(len(eigvals_prod), n_samples))

        # compute the null distribution consisting now of (n_samples)
        # of chi-squared random variables weighted by the eigenvalue products
        null_dist = 1.0 / n * eigvals_prod.T.dot(f_rand)
        return null_dist

    def _compute_null_ci(self, uu_prod, n_samples):
        # the eigenvalues of ww^T
        eig_uu = np.linalg.eigvalsh(uu_prod)
        eig_uu = eig_uu[eig_uu > eig_uu.max() * self.threshold]

        # generate chi-square distributed values z_{ij} with degree of freedom 1
        f_rand = np.random.chisquare(df=1, size=(eig_uu.shape[0], n_samples))

        # compute the null distribution consisting now of (n_samples)
        # of chi-squared random variables weighted by the eigenvalue products
        null_dist = eig_uu.T.dot(f_rand)
        return null_dist

    def _compute_kernel(
        self, X, Y=None, distance_metric="l2", metric="rbf", kwidth=None, centered=True
    ):
        # if the width of the kernel is not set, then use the median trick to set the
        # kernel width based on the data X
        if kwidth is None:
            # Note: sigma = 1 / np.sqrt(kwidth)
            # compute N x N pairwise distance matrix
            dists = pairwise_distances(X, metric=distance_metric, n_jobs=self.n_jobs)

            # compute median of off diagonal elements
            med = np.median(dists[dists > 0])

            # prevents division by zero when used on label vectors
            med = med if med else 1
        else:
            med = kwidth

        extra_kwargs = dict()

        if metric == "rbf":
            # compute the normalization factor of the width of the Gaussian kernel
            gamma = 1.0 / (2 * (med**2))
            extra_kwargs["gamma"] = gamma
        elif metric == "polynomial":
            degree = 2
            extra_kwargs["degree"] = degree

        # compute the potentially pairwise kernel
        kernel = pairwise_kernels(X, Y=Y, metric=metric, n_jobs=self.n_jobs, **extra_kwargs)

        if centered:
            kernel = self._center_kernel(kernel)
        return kernel, med

    def _center_kernel(self, K):
        """Centers the kernel matrix.

        Applies a transformation H * K * H, where H is a diagonal matrix with 1/n along
        the diagonal.
        """
        n = K.shape[0]
        H = np.eye(n) - 1.0 / n
        return H.dot(K).dot(H)
