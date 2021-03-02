#  Copyright (c) 2019 École Polytechnique
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
#
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from aideme.utils import assert_positive


class LaplaceBayesianLogisticRegressionBase:
    GAUSSIAN_CDF_APPROX_FACTOR = 8 / np.pi

    def __init__(self, prior: str = 'gaussian', prior_std: float = 1.0, add_intercept: bool = False, tol: float = 1e-6, max_iter: int = 10000):
        """
        :param prior: prior for logistic regression weights. Available options are 'gaussian' and 'improper'
        :param prior_std: standard deviation of prior distribution. It has no effect for 'improper' prior.
        :param max_iter: int, default=10000
            Maximum number of iterations for the solver.
        :param tol: float, default=1e-6
            Stopping criterion. The iteration will stop when ``max{|g_i | i = 1, ..., n} <= tol``,
             where ``g_i`` is the i-th component of the gradient.

        """
        assert_positive(prior_std, 'prior_std')
        assert_positive(tol, 'tol')
        assert_positive(max_iter, 'max_iter')

        self._add_intercept = add_intercept
        self._info = self.__get_info(prior, prior_std)
        self._standard_normal_dist = norm(0, 1)

        self._mean = None
        self._L = None

        self._opt_options = {
            'gtol': tol,
            'maxiter': max_iter,
            'maxfun': max_iter,
        }

    @staticmethod
    def __get_info(prior: str, prior_std: float) -> float:
        if prior == 'improper':
            return 0.
        if prior == 'gaussian':
            return 1 / (prior_std * prior_std)
        raise ValueError("Unknown prior option: {}".format(prior))

    def clear(self) -> None:
        self._mean = None
        self._L = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.__add_intercept(X)

        mu = X.dot(self._mean)

        var = self._compute_variance(X)
        var += self.GAUSSIAN_CDF_APPROX_FACTOR

        return self._standard_normal_dist.cdf(mu / np.sqrt(var))

    def _compute_variance(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Sample from the posterior distribution through an MCMC sampler.
        :param X: data matrix
        :param y: labels (positive label should be 1)
        :return: samples in a numpy array (one per line)
        """
        X = self.__add_intercept(X)

        self._mean = self._compute_map_estimator(X, y)

        H = self._compute_hessian(X, self._mean)
        self._L = cholesky(H)

    def _compute_map_estimator(self, X, y):
        raise NotImplementedError

    def _minimize(self, dim, loss, args):
        w0 = np.zeros(dim)
        res = minimize(
            loss, w0, method='L-BFGS-B', jac=True,
            args=args,
            options=self._opt_options
        )

        if not res.success:
            raise RuntimeError('Optimization failed: MAP estimator could not be computed.')

        return res.x

    def _compute_hessian(self, X, w):
        raise NotImplementedError

    def __add_intercept(self, X):
        if self._add_intercept:
            ones = np.ones(shape=(len(X), 1))
            X = np.hstack([ones, X])

        return X


class LaplaceBayesianLogisticRegression(LaplaceBayesianLogisticRegressionBase):
    def _compute_variance(self, X: np.ndarray) -> np.ndarray:
        L_inv_X = solve_triangular(self._L, X.T, lower=True, trans=0)
        var = np.square(L_inv_X).sum(axis=0)
        return var

    def _compute_map_estimator(self, X, y):
        A = X * np.where(y == 1, -1, 1).reshape(-1, 1)
        return self._minimize(A.shape[1], self._loss, args=(A,))

    def _loss(self, w, A):
        m = A.dot(w)
        loss = np.logaddexp(0, m).sum() + 0.5 * self._info * w.dot(w)
        grad = A.T.dot(expit(m)) + self._info * w
        return loss, grad

    def _compute_hessian(self, X, w):
        P = expit(X.dot(w))
        P = P * (1 - P)
        H = np.einsum('i,ir,is -> rs', P, X, X)
        H[np.diag_indices_from(H)] += max(1e-12, self._info)
        return H


class KernelLaplaceBayesianLogisticRegression(LaplaceBayesianLogisticRegressionBase):
    def __init__(self, prior_std: float = 1.0, tol: float = 1e-6, max_iter: int = 10000):
        super().__init__(prior='gaussian', prior_std=prior_std, add_intercept=False, tol=tol, max_iter=max_iter)

    def _compute_variance(self, K_test: np.ndarray) -> np.ndarray:
        inner = solve_triangular(self._L, (K_test * self.C).T, lower=True, trans=0)
        inner = np.square(inner).sum(axis=0)
        var = (1 - inner) / self._info
        return var

    def _compute_map_estimator(self, K, y):
        y = np.where(y == 1, -1., 1.)
        return self._minimize(K.shape[1], self._loss, args=(K, y))

    def _loss(self, w, K, y):
        prod = K.dot(w)  # <k_i, alpha>
        m = prod * y  # -y_i * <k_i, alpha>
        loss = np.logaddexp(0, m).sum() + 0.5 * self._info * w.dot(prod)
        grad = K.T.dot(y * expit(m)) + self._info * prod
        return loss, grad

    def _compute_hessian(self, K, w):
        self.C = expit(K.dot(w))
        self.C *= (1 - self.C)
        np.sqrt(self.C, out=self.C)

        K *= self.C
        K *= self.C.reshape(-1, 1)
        K[np.diag_indices_from(K)] += self._info

        return K
