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
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from aideme.utils import assert_positive


class ApproximateBayesianLogisticRegression:

    GAUSSIAN_CDF_APPROX_FACTOR = 8 / np.pi

    def __init__(self, prior: str = 'gaussian', prior_std: float = 1.0, add_intercept: bool = True, tol=1e-12, max_iter=10000):
        """
        :param prior: prior for logistic regression weights. Available options are 'gaussian' and 'improper'
        :param prior_std: standard deviation of prior distribution. It has no effect for 'improper' prior.
        :param max_iter: int, default=10000
            Maximum number of iterations for the solver.
        :param tol: float, default=1e-12
            Stopping criterion. The iteration will stop when ``max{|g_i | i = 1, ..., n} <= tol``,
             where ``g_i`` is the i-th component of the gradient.

        """
        assert_positive(prior_std, 'prior_std')
        assert_positive(tol, 'tol')
        assert_positive(max_iter, 'max_iter')

        self.add_intercept = add_intercept
        self.info = self.__get_info(prior, prior_std)
        self.standard_normal_dist = norm(0, 1)

        self.mean = None
        self.L = None

        self.opt_options = {
            'gtol': tol,
            'maxiter': max_iter
        }

    @staticmethod
    def __get_info(prior: str, prior_std: float) -> float:
        if prior == 'improper':
            return 0.
        if prior == 'gaussian':
            return 1 / (prior_std * prior_std)
        raise ValueError("Unknown prior option: {}".format(prior))

    def clear(self) -> None:
        self.mean = None
        self.L = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.add_intercept:
            ones = np.ones(shape=(len(X), 1))
            X = np.hstack([ones, X])

        # approximate inference: p = Phi(mu / (8/pi + sig^2)), where mu = x^T w and sig^2 = x^T H^-1 x = || L^-1 x ||^2
        mu = X.dot(self.mean)

        L_inv_X = solve_triangular(self.L, X.T, lower=True, trans=0)
        var = np.square(L_inv_X).sum(axis=0)
        var += self.GAUSSIAN_CDF_APPROX_FACTOR

        return self.standard_normal_dist.cdf(mu / np.sqrt(var))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Sample from the posterior distribution through an MCMC sampler.
        :param X: data matrix
        :param y: labels (positive label should be 1)
        :return: samples in a numpy array (one per line)
        """
        if self.add_intercept:
            ones = np.ones(shape=(len(X), 1))
            X = np.hstack([ones, X])

        A = X * np.where(y == 1, -1, 1).reshape(-1, 1)

        # compute gaussian approximation mean and (inverse) covariance matrix
        self.mean = self._compute_map_estimator(A)

        # compute Cholesky decomposition of H
        H = self._compute_hessian(self.mean, X)
        self.L = np.linalg.cholesky(H)

    def _compute_map_estimator(self, A):
        w0 = np.zeros(A.shape[1])
        res = minimize(
            self._loss, w0, method='L-BFGS-B', jac=True,
            args=(A,),
            options=self.opt_options
        )

        if not res.success:
            raise RuntimeError('Optimization failed: MAP estimator could not be computed.')

        return res.x

    def _loss(self, w, A):
        m = A.dot(w)
        loss = np.logaddexp(0, m).sum() + 0.5 * self.info * w.dot(w)
        grad = A.T.dot(expit(m)) + self.info * w
        return loss, grad

    def _compute_hessian(self, w, X):
        P = expit(X.dot(w))
        P = P * (1 - P)
        H = np.einsum('i,ir,is -> rs', P, X, X)
        H[np.diag_indices_from(H)] += max(1e-12, self.info)
        return H
