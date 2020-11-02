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

import warnings
from typing import Generator, List, Optional, Tuple

import numpy as np
from scipy.special import expit

from .gradient_descent import GradientDescentOptimizer
from .utils import log1mexp


class FactorizedLinearClassifier:
    __LOGHALF = np.log(0.5)

    def __init__(self, partition: List[List[int]], weights: Optional[np.ndarray] = None):
        self.partition = partition
        self._dim = len(partition) + sum(map(len, partition))

        if weights is None:
            weights = np.zeros(self._dim)

        if len(weights) != self._dim:
            raise ValueError("Incompatible dimension for weights: expected {}, but got {}".format(self._dim, len(weights)))

        self.weights = weights
        self.__offsets = np.cumsum([len(p) + 1 for p in self.partition])

    @property
    def dim(self) -> int:
        return self._dim

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self._log_proba(X) > self.__LOGHALF, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._log_proba(X))

    def partial_proba(self, X: np.ndarray, i: int) -> np.ndarray:
        begin = self.__offsets[i-1] if i > 0 else 0
        end = self.__offsets[i]
        return self._logistic_proba(X, self.weights[begin:end])

    def merge_partitions(self, i: int, j: int, merge_weights: bool = True) -> FactorizedLinearClassifier:
        if i == j:
            return self

        i, j = min(i, j), max(i, j)

        merged_part = (
                self.partition[:i]
                + self.partition[i + 1:j]
                + self.partition[j + 1:]
                + [self.partition[i] + self.partition[j]]
        )

        if not merge_weights:
            return FactorizedLinearClassifier(merged_part)

        begin_i, end_i = self.__offsets[i - 1] if i > 0 else 0, self.__offsets[i]
        begin_j, end_j = self.__offsets[j - 1], self.__offsets[j]

        merged_w = np.hstack([
            self.weights[:begin_i],
            self.weights[end_i:begin_j],
            self.weights[end_j:],
            [self.weights[begin_i] + self.weights[begin_j]],
            self.weights[begin_i + 1:end_i],
            self.weights[begin_j + 1:end_j]
        ])

        return FactorizedLinearClassifier(merged_part, merged_w)

    def _log_proba(self, X: np.ndarray) -> np.ndarray:
        log_probas = np.zeros(len(X), dtype='float')

        for X_partial, w_partial in self.__generate_subspace_data(X):
            partial_proba = self._logistic_proba(X_partial, w_partial)
            log_probas += np.log(partial_proba)

        return log_probas

    def _grad_log_proba(self, X: np.ndarray) -> np.ndarray:
        grads = np.ones((self.dim, len(X)))

        for begin, end, X_partial, w_partial in self.__generate_subspace_data(X, include_endpoints=True):
            partial_proba = self._logistic_proba(X_partial, w_partial)

            grads[begin+1:end] = X_partial.T
            grads[begin:end] *= (1 - partial_proba)

        return grads

    def __generate_subspace_data(self, X: np.ndarray, include_endpoints: bool = False) -> Generator:
        begin = 0

        for p, end in zip(self.partition, self.__offsets):
            if include_endpoints:
                yield begin, end, X[:, p], self.weights[begin:end]
            else:
                yield X[:, p], self.weights[begin:end]
            begin = end

    @staticmethod
    def _logistic_proba(X, w):
        return expit(w[0] + X.dot(w[1:]))


class FactorizedLinearLearner:
    def __init__(self, optimizer: str = 'GD', **opt_params):
        self.__opt = self.__get_optimizer(optimizer, opt_params)

    @staticmethod
    def __get_optimizer(optimizer: str, opt_params):
        # TODO: allow for other optimization methods?
        optimizer = optimizer.upper()

        if optimizer == 'GD':
            return GradientDescentOptimizer(**opt_params)

        raise ValueError("Unknown optimizer: {}".format(optimizer))

    def fit(self, X: np.ndarray, y: np.ndarray, partition: List[List[int]], x0: Optional[np.ndarray] = None):
        return self.fit_and_loss(X, y, partition, x0)[0]

    def compute_factorization_loss(self, X: np.ndarray, y: np.ndarray, partition: List[List[int]], x0: Optional[np.ndarray] = None):
        return self.fit_and_loss(X, y, partition, x0)[1]

    def fit_and_loss(self, X: np.ndarray, y: np.ndarray, partition: List[List[int]], x0: Optional[np.ndarray] = None) -> Tuple[FactorizedLinearClassifier, float]:
        fact_clf = FactorizedLinearClassifier(partition, x0)
        loss = LinearLoss(X, y, fact_clf)

        res = self.__opt.optimize(loss.x0, loss.loss, loss.grad)

        if not res.converged:
            warnings.warn("Optimization routine did not converge.\n{}".format(res))

        fact_clf.weights = res.x

        return fact_clf, res.fun


class LinearLoss:
    def __init__(self, X: np.ndarray, y: np.ndarray, fact_clf: FactorizedLinearClassifier):
        self.X = X
        self.is_positive = y > 0
        self.fact_clf = fact_clf
        self.x0 = self.fact_clf.weights.copy()

    def loss(self, weights: np.ndarray) -> np.ndarray:
        self.fact_clf.weights = weights

        log_probas = self.fact_clf._log_proba(self.X)
        return -np.where(self.is_positive, log_probas, log1mexp(log_probas)).mean()

    def grad(self, weights: np.ndarray) -> np.ndarray:
        self.fact_clf.weights = weights

        log_probas = self.fact_clf._log_proba(self.X)

        grads = self.fact_clf._grad_log_proba(self.X)
        with warnings.catch_warnings():  # TODO: can we avoid this? Write in Cython?
            warnings.simplefilter("ignore")
            weights = np.where(self.is_positive, -1, 1 / np.expm1(-log_probas))
        return grads.dot(weights) / len(self.X)
