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
from typing import Optional

import numpy as np
import scipy.optimize

import aideme.active_learning.factorization.utils as utils
from aideme.utils import assert_non_negative, assert_positive_integer
from .gradient_descent import ProximalGradientDescentOptimizer, l1_penalty


class LinearFactorizationLearner:
    def __init__(self, add_bias: bool = True, interaction_penalty: float = 0, l1_penalty: float = 0,
                 huber_penalty: float = 0, huber_delta: float = 1e-3):
        assert_non_negative(interaction_penalty, 'interaction_penalty')
        assert_non_negative(huber_penalty, 'huber_penalty')
        assert_non_negative(huber_delta, 'huber_delta')

        self.add_bias = add_bias
        self.interaction_penalty = interaction_penalty
        self.l1_penalty = l1_penalty
        self.huber_penalty = huber_penalty
        self.huber_delta = huber_delta

        self._weights = None
        self._bias = 0

        self._optimizer = self.__get_optimizer()

    def __get_optimizer(self):
        if self.l1_penalty > 0:
            optimizer = ProximalGradientDescentOptimizer()
            g, prox = l1_penalty(self.l1_penalty, self.add_bias)
            # TODO: optimize f(x), fprime(x) computation
            return lambda x0, func: optimizer.minimize(x0, lambda x: func(x)[0], lambda x: func(x, return_matrix=True)[1], g, prox)

        return lambda x0, func: scipy.optimize.minimize(func, x0, jac=True, method='bfgs')

    def fit(self, X: np.ndarray, y: np.ndarray, max_partitions: int, x0: Optional[np.ndarray] = None):
        assert_positive_integer(max_partitions, 'max_partitions')

        loss = LinearFactorizationLoss(X=X, y=y, add_bias=self.add_bias,
                                       interaction_penalty=self.interaction_penalty,
                                       huber_penalty=self.huber_penalty, huber_delta=self.huber_delta)

        shape = (max_partitions, loss.X.shape[1])

        if x0 is None:
            x0 = np.random.uniform(-1, 1, size=shape)

        opt_result = self._optimizer(x0, loss)

        self._weights = opt_result.x.reshape(shape)
        if self.add_bias:
            self._bias = self._weights[:, -1]
            self._weights = self._weights[:, :-1]

        return opt_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probas = utils.compute_log_probas(self._margin(X))
        return np.where(log_probas > np.log(0.5), 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probas = utils.compute_log_probas(self._margin(X))
        return np.exp(log_probas)

    def _margin(self, X: np.ndarray) -> np.ndarray:
        return X @ self._weights.T + self._bias


class LinearFactorizationLoss:
    def __init__(self, X: np.ndarray, y: np.ndarray, add_bias: bool = True, interaction_penalty: float = 0,
                 huber_penalty: float = 0, huber_delta: float = 1e-3):
        if add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        self.X = X
        self.y = y
        self.add_bias = add_bias
        self.interaction_penalty = interaction_penalty
        self.huber_penalty = huber_penalty
        self.huber_delta = huber_delta

    def __call__(self, weights: np.ndarray, return_matrix: bool = False):
        weights = weights.reshape(-1, self.X.shape[1])

        margins = self.X @ weights.T
        loss, grad_weights = utils.compute_loss_and_grad(margins, self.y)
        grads = grad_weights.T @ self.X

        if self.interaction_penalty > 0:
            loss += self.__add_penalty(self.compute_interaction_penalty, weights, grads)

        if self.huber_penalty > 0:
            loss += self.__add_penalty(self.compute_huber_penalty, weights, grads)

        return loss, (grads if return_matrix else grads.ravel())

    def compute_interaction_penalty(self, weights):
        if self.add_bias:
            weights = weights[:, :-1]

        M = weights @ weights.T
        np.fill_diagonal(M, 0)

        penalty = self.interaction_penalty * np.square(M).sum()
        penalty_grad = (4 * self.interaction_penalty) * (M @ weights)
        return penalty, penalty_grad

    def compute_huber_penalty(self, weights):
        return utils.compute_huber_loss_and_grad(weights, self.huber_penalty, self.huber_delta, int(self.add_bias))

    def __add_penalty(self, penalty_func, weights, grads):
        penalty_loss, grads_loss = penalty_func(weights)

        idx = grads.shape[1] - self.add_bias
        grads[:, :idx] += grads_loss

        return penalty_loss
