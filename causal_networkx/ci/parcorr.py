import math
import sys

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


class PartialCorrelation:
    def __init__(self, random_state=None, method="analytic", fixed_threshold=0.1, **kwargs):
        self._measure = "par_corr"
        self.method = method
        self.two_sided = True
        self.residual_based = True
        self.fixed_threshold = fixed_threshold
        self.random_state = random_state

    def test(self, X, Y, Z=None):
        """Perform CI test of X, Y given optionally Z.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_dimensions)
            The first dataset.
        Y : np.ndarray of shape (n_samples, n_dimensions)
            The dataset for Y.
        Z : np.ndarray of shape (n_samples, n_dimensions), optional
            The conditioning dataset, by default None.
        """
        # stack the X and Y arrays
        XY = np.hstack((X, Y))
        xdim = X.shape[1]
        ydim = Y.shape[1]

        xy_ind = np.array([0 for i in range(xdim)] + [1 for i in range(ydim)])

        # handle if conditioning set is passed in or not
        if Z is None:
            data_arr = XY
            data_ind = xy_ind
        else:
            zdim = Z.shape[1]
            data_arr = np.hstack((XY, Z))
            data_ind = np.array(
                [0 for i in range(xdim)] + [1 for i in range(ydim)] + [2 for i in range(zdim)]
            )

        # Ensure it is a valid array
        if np.isnan(data_arr).sum() != 0:
            raise ValueError("nans in the data array.")

        #
        n_samples, n_dims = data_arr.shape

        # compute the dependence measure of the data vs indicator function
        val = self._compute_parcorr(data_arr, xvalue=0, yvalue=1)

        # compute the pvalue
        pvalue = self.compute_significance(val, data_arr, data_ind, n_samples, n_dims)

        return val, pvalue

    def compute_significance(self, val, array, xyz, n_samples, n_dims, sig_override=None):
        """
        Returns the p-value from whichever significance function is specified
        for this test.  If an override is used, then it will call a different
        function then specified by self.significance

        Parameters
        ----------
        val : float
            Test statistic value.

        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        n_samples : int
            Sample length

        n_dims : int
            Dimensionality, ie, number of features.

        sig_override : string
            Must be in 'analytic', 'shuffle_test', 'fixed_thres'

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Defaults to the self.significance member value
        use_sig = self.method
        if sig_override is not None:
            use_sig = sig_override

        # Check if we are using the analytic significance
        if use_sig == "analytic":
            pval = self._compute_analytic_significance(
                value=val, n_samples=n_samples, n_dims=n_dims
            )
        # Check if we are using the shuffle significance
        elif use_sig == "shuffle_test":
            pval = self._compute_shuffle_significance(array=array, xyz=xyz, value=val)
        # Check if we are using the fixed_thres significance
        elif use_sig == "fixed_thresh":
            pval = self._compute_fixed_threshold_significance(
                value=val, fixed_threshold=self.fixed_threshold
            )
        else:
            raise ValueError("%s not known." % self.method)

        # Return the calculated value
        return pval

    def _compute_parcorr(self, array, xvalue, yvalue):
        # compute residuals when regressing Z on X and Z on Y
        x_resid = self._compute_ols_residuals(array, target_var=xvalue)
        y_resid = self._compute_ols_residuals(array, target_var=yvalue)

        # then compute the correlation using Pearson method
        val, _ = stats.pearsonr(x_resid, y_resid)
        return val

    def _compute_ols_residuals(self, array, target_var, standardize=True, return_means=False):
        """Compute residuals of linear multiple regression.

        Performs a OLS regression of the variable indexed by ``target_var`` on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.

        Parameters
        ----------
        array : np.ndarray of shape (n_samples, n_vars)
            Data array with X, Y, Z in rows and observations in columns.

        target_var : int
            The variable to regress out conditions from. This should
            be the value of the X or Y indicator (0, or 1) in this case
            indicating the row in ``array`` for those two datas.

        standardize : bool, optional
            Whether to standardize the array beforehand. Must be used for
            partial correlation. Default is True.

        return_means : bool, optional
            Whether to return the estimated regression line. Default is False.

        Returns
        -------
        resid : np.ndarray of shape
            The residual of the regression and optionally the estimated line.
        mean : np.ndarray of shape
        """
        T, dim = array.shape
        dim_z = dim - 2

        # standardize with z-score transformation
        if standardize:
            scaler = StandardScaler()
            array = scaler.fit_transform(array)

        y = array[:, target_var]

        if dim_z > 0:
            # get the (n_samples, zdim) array
            z = np.fastCopyAndTranspose(array[:, 2:])

            # compute the least squares regression of z @ \beta = y
            # - y is a (n_samples, ydim) array
            # - beta is a (zdim, ydim) array of values
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]

            # compute the residuals of the model predictions vs actual values
            y_pred = np.dot(z, beta_hat)
            resid = y - y_pred
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        return resid

    def _compute_shuffle_significance(self, array, xyz, value, return_null_dist=False):
        """Returns p-value for shuffle significance test.

        For residual-based test statistics only the residuals are shuffled.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """

        x_vals = self._compute_ols_residuals(array, target_var=0)
        y_vals = self._compute_ols_residuals(array, target_var=1)
        array_resid = np.array([x_vals, y_vals])
        xyz_resid = np.array([0, 1])

        null_dist = self._get_shuffle_dist(
            array_resid,
            xyz_resid,
            self.get_dependence_measure,
            sig_samples=self.sig_samples,
            sig_blocklength=self.sig_blocklength,
            verbosity=self.verbosity,
        )

        pval = (null_dist >= np.abs(value)).mean()

        # Adjust p-value for two-sided measures
        if pval < 1.0:
            pval *= 2.0

        if return_null_dist:
            return pval, null_dist
        return pval

    def _get_shuffle_dist(
        self, array, xyz, dependence_measure, sig_samples, sig_blocklength=None, verbosity=0
    ):
        """Returns shuffle distribution of test statistic.

         The rows in array corresponding to the X-variable are shuffled using
         a block-shuffle approach.

         Parameters
         ----------
         array : array-like
             data array with X, Y, Z in rows and observations in columns

         xyz : array of ints
             XYZ identifier array of shape (dim,).

        dependence_measure : object
            Dependence measure function must be of form
            dependence_measure(array, xyz) and return a numeric value

         sig_samples : int, optional (default: 100)
             Number of samples for shuffle significance test.

         sig_blocklength : int, optional (default: None)
             Block length for block-shuffle significance test. If None, the
             block length is determined from the decay of the autocovariance as
             explained in [1]_.

         verbosity : int, optional (default: 0)
             Level of verbosity.

         Returns
         -------
         null_dist : array of shape (sig_samples,)
             Contains the sorted test statistic values estimated from the
             shuffled arrays.
        """

        dim, T = array.shape

        x_indices = np.where(xyz == 0)[0]
        dim_x = len(x_indices)

        if sig_blocklength is None:
            sig_blocklength = self._get_block_length(array, xyz, mode="significance")

        n_blks = int(math.floor(float(T) / sig_blocklength))
        # print 'n_blks ', n_blks
        if verbosity > 2:
            print("            Significance test with block-length = %d " "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blks * sig_blocklength :]

        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):

            blk_starts = self.random_state.permutation(block_starts)[:n_blks]

            x_shuffled = np.zeros((dim_x, n_blks * sig_blocklength), dtype=array.dtype)

            for i, index in enumerate(x_indices):
                for blk in range(sig_blocklength):
                    x_shuffled[i, blk::sig_blocklength] = array[index, blk_starts + blk]

            # Insert tail randomly somewhere
            if tail.shape[1] > 0:
                insert_tail_at = self.random_state.choice(block_starts)
                x_shuffled = np.insert(x_shuffled, insert_tail_at, tail.T, axis=1)

            for i, index in enumerate(x_indices):
                array_shuffled[index] = x_shuffled[i]

            null_dist[sam] = dependence_measure(array=array_shuffled, xyz=xyz)

        return null_dist

    def _compute_analytic_significance(self, value, n_samples, n_dims):
        """Analytic p-value from Student's t-test for Pearson correlation coefficient.

        Assumes two-sided correlation. If the degrees of freedom are less than
        1, numpy.nan is returned.

        Parameters
        ----------
        value : float
            Test statistic value.
        n_samples : int
            Sample length
        n_dims : int
            Dimensionality, ie, number of features.

        Returns
        -------
        pval : float | numpy.nan
            P-value.
        """
        # Get the number of degrees of freedom
        deg_f = n_samples - n_dims

        if deg_f < 1:
            pval = np.nan
        elif abs(abs(value) - 1.0) <= sys.float_info.min:
            pval = 0.0
        else:
            trafo_val = value * np.sqrt(deg_f / (1.0 - value * value))
            # Two sided significance level
            pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2

        return pval

    def _compute_fixed_threshold_significance(self, value, fixed_threshold):
        """Returns signficance for thresholding test.

        Returns 0 if numpy.abs(value) is smaller than ``fixed_threshold`` and 1 else.

        Parameters
        ----------
        value : float
            Value of test statistic for unshuffled estimate.

        fixed_threshold : float
            Fixed threshold, is made positive.

        Returns
        -------
        pval : float
            Returns 0 if numpy.abs(value) is smaller than fixed_thres and 1
            else.
        """
        if np.abs(value) < np.abs(fixed_threshold):
            pval = 1.0
        else:
            pval = 0.0

        return pval
