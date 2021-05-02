#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 05:17:25 2020

@author: marty
"""

import numpy as np
from numba import int32, float32, jit
from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, dot, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones, cov, linalg)
from scipy import special
from scipy.stats import mvn
import warnings
from sklearn.utils import check_random_state
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.special import kl_div

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jit(cache=True)
def gaussian_kernel_estimate(points, values, xi, precision, dtype, real=0):
    """
    def gaussian_kernel_estimate(points, real[:, :] values, xi, precision)

    Evaluate a multivariate Gaussian kernel estimate.

    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimenions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    precision : array_like with shape (d, d)
        Precision matrix for the Gaussian kernel.

    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    # cdef:
    #     real[:, :] points_, xi_, values_, estimate, whitening
    #     int i, j, k
    #     int n, d, m, p
    #     real arg, residual, norm

    n, d = points.shape
    m, p = xi.shape

    # if xi.shape[1] != d:
    #     raise ValueError("points and xi must have same trailing dim")
    # if precision.shape[0] != d or precision.shape[1] != d:
    #     raise ValueError("precision matrix must match data dims")

    # Rescale the data
    whitening = np.linalg.cholesky(precision).astype(dtype, copy=False)
    points_ = np.dot(points, whitening).astype(dtype, copy=False)
    xi_ = np.dot(xi, whitening).astype(dtype, copy=False)
    values_ = values.astype(dtype, copy=False)

    # Evaluate the normalisation
    norm = (2 * pi) ** (- d / 2)
    for i in range(d):
        norm *= whitening[i, i]

    # Create the result array and evaluate the weighted sum
    estimate = np.zeros((m, p), dtype)
    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])
                arg += residual * residual

            arg = np.exp(-arg / 2) * norm
            for k in range(p):
                estimate[j, k] += values_[i, k] * arg

    return np.asarray(estimate)

class new_gaussian_kde(object):
    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = atleast_2d(asarray(dataset))
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self._weights = atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1/sum(self._weights**2)

        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """
        Evaluate the estimated pdf on a provided set of points.
        """
        points = atleast_2d(asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        output_dtype = np.common_type(self.covariance, points)
        itemsize = np.dtype(output_dtype).itemsize
        if itemsize == 4:
            spec = 'float'
        elif itemsize == 8:
            spec = 'double'
        elif itemsize in (12, 16):
            spec = 'long double'
        else:
            raise TypeError('%s has unexpected item size %d' %
                            (output_dtype, itemsize))
        result = gaussian_kernel_estimate(self.dataset.T, self.weights[:, None],
                                                points.T, self.inv_cov, output_dtype)
        return result[:, 0]

    __call__ = evaluate

    def integrate_gaussian(self, mean, cov):
        mean = atleast_1d(squeeze(mean))
        cov = atleast_2d(cov)

        if mean.shape != (self.d,):
            raise ValueError("mean does not have dimension %s" % self.d)
        if cov.shape != (self.d, self.d):
            raise ValueError("covariance does not have dimension %s" % self.d)

        # make mean a column vector
        mean = mean[:, newaxis]

        sum_cov = self.covariance + cov

        # This will raise LinAlgError if the new cov matrix is not s.p.d
        # cho_factor returns (ndarray, bool) where bool is a flag for whether
        # or not ndarray is upper or lower triangular
        sum_cov_chol = linalg.cho_factor(sum_cov)

        diff = self.dataset - mean
        tdiff = linalg.cho_solve(sum_cov_chol, diff)

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        energies = sum(diff * tdiff, axis=0) / 2.0
        result = sum(exp(-energies)*self.weights, axis=0) / norm_const

        return result

    def integrate_box_1d(self, low, high):
        if self.d != 1:
            raise ValueError("integrate_box_1d() only handles 1D pdfs")

        stdev = ravel(sqrt(self.covariance))[0]

        normalized_low = ravel((low - self.dataset) / stdev)
        normalized_high = ravel((high - self.dataset) / stdev)

        value = np.sum(self.weights*(
                        special.ndtr(normalized_high) -
                        special.ndtr(normalized_low)))
        return value

    def integrate_box(self, low_bounds, high_bounds, maxpts=None):
        if maxpts is not None:
            extra_kwds = {'maxpts': maxpts}
        else:
            extra_kwds = {}

        value, inform = mvn.mvnun_weighted(low_bounds, high_bounds,
                                           self.dataset, self.weights,
                                           self.covariance, **extra_kwds)
        if inform:
            msg = ('An integral in mvn.mvnun requires more points than %s' %
                   (self.d * 1000))
            warnings.warn(msg)

        return value

    def integrate_kde(self, other):
        if other.d != self.d:
            raise ValueError("KDEs are not the same dimensionality")

        # we want to iterate over the smallest number of points
        if other.n < self.n:
            small = other
            large = self
        else:
            small = self
            large = other

        sum_cov = small.covariance + large.covariance
        sum_cov_chol = linalg.cho_factor(sum_cov)
        result = 0.0
        for i in range(small.n):
            mean = small.dataset[:, i, newaxis]
            diff = large.dataset - mean
            tdiff = linalg.cho_solve(sum_cov_chol, diff)

            energies = sum(diff * tdiff, axis=0) / 2.0
            result += sum(exp(-energies)*large.weights, axis=0)*small.weights[i]

        sqrt_det = np.prod(np.diagonal(sum_cov_chol[0]))
        norm_const = power(2 * pi, sum_cov.shape[0] / 2.0) * sqrt_det

        result /= norm_const

        return result

    def resample(self, size=None, seed=None):
        if size is None:
            size = int(self.neff)

        random_state = check_random_state(seed)
        norm = transpose(random_state.multivariate_normal(
            zeros((self.d,), float), self.covariance, size=size
        ))
        indices = random_state.choice(self.n, size=size, p=self.weights)
        means = self.dataset[:, indices]

        return means + norm

    def scotts_factor(self):
        return power(self.neff, -1./(self.d+4))

    def silverman_factor(self):
        return power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = """Computes the coefficient (`kde.factor`) that
        multiplies the data covariance matrix to obtain the kernel covariance
        matrix. The default is `scotts_factor`.  A subclass can overwrite this
        method to provide a different method, or set it through a call to
        `kde.set_bandwidth`."""

    def set_bandwidth(self, bw_method=None):
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(2*pi*self.covariance))

    def pdf(self, x):
        """
        Evaluate the estimated pdf on a provided set of points.
        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.
        """
        return self.evaluate(x)

    def logpdf(self, x):
        """
        Evaluate the log of the estimated pdf on a provided set of points.
        """

        points = atleast_2d(x)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        if m >= self.n:
            # there are more points than data, so loop over data
            energy = np.empty((self.n, m), dtype=float)
            for i in range(self.n):
                diff = self.dataset[:, i, newaxis] - points
                tdiff = dot(self.inv_cov, diff)
                energy[i] = sum(diff*tdiff, axis=0) / 2.0
            result = logsumexp(-energy.T,
                               b=self.weights / self._norm_factor, axis=1)
        else:
            # loop over points
            result = np.empty((m,), dtype=float)
            for i in range(m):
                diff = self.dataset - points[:, i, newaxis]
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff * tdiff, axis=0) / 2.0
                result[i] = logsumexp(-energy, b=self.weights /
                                      self._norm_factor)

        return result

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = ones(self.n)/self.n
            return self._weights

    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1/sum(self.weights**2)
            return self._neff

def ave_kl_div(X, new_X, x_range, plotPDF=True, printKLVal=True, plotIndex=True):
    '''
    Takes two datasets and returns the Kullback Leibler divergence for each column

    Parameters
    ----------
    X : 2D ndarray
        Theoretical time series.
    new_X : 2D ndarray
        Predicted time series.
    x_range : 1D ndarray
        The range of values over which you want to compute the KL divergence.
    plotPDF : Boolean
        Plots the PDFs for each column if True
    printKLVal : Boolean
        Prints the KL_div for each column
    plotIndex : Boolean
        Plots KL_div vs Index points of True

    Returns
    -------
    kl_vals : Float
        The Average Kullback Leibler divergence.

    '''
    _,K = X.shape
    kl_vals = np.zeros(K)
    for i in range(K):
        data1 = X[:, i]
        density1 = new_gaussian_kde(data1, "scott")
        data2 = new_X[:, i]
        density2 = new_gaussian_kde(data2, "scott")
        val1 = density1(x_range)
        val2 = density2(x_range)
        if plotPDF:
            plt.plot(x_range, val1)
            plt.plot(x_range, val2)
            plt.legend(["Data", "Prediction"])
            plt.title("X_"+str(i+1))
            plt.show()
        kl_vals[i] = np.sum([kl_div(val1, val2)])
        if printKLVal:
            print("X_"+str(i+1)+", Kullback-Leibler Divergence: "+str(np.round(kl_vals[i], 5)))
    if plotIndex:
        plt.plot(list(range(1,K+1)), kl_vals, "--o")
        plt.plot(list(range(1,K+1)), np.mean(kl_vals)*np.ones(K), "--r")
        plt.xlabel("Slow variable index")
        plt.ylabel("K-L Divergence")
        plt.legend(["K-L Divergence", "Average K-L Divergence"])
        plt.show()
    return np.mean(kl_vals)