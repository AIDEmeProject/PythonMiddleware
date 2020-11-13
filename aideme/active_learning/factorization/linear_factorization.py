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

from typing import Optional, List, Union

import numpy as np
import scipy.optimize

import aideme.active_learning.factorization.utils as utils
from aideme.utils import assert_non_negative, assert_positive, assert_positive_integer
from .gradient_descent import ProximalGradientDescentOptimizer, l1_penalty_func_and_prox


class LinearFactorizationLearner:
    def __init__(self, add_bias: bool = True, interaction_penalty: float = 0, l1_penalty: float = 0,
                 huber_penalty: float = 0, huber_delta: float = 1e-3, tol: float = 1e-4):
        assert_non_negative(interaction_penalty, 'interaction_penalty')
        assert_non_negative(l1_penalty, 'l1_penalty')
        assert_non_negative(huber_penalty, 'huber_penalty')
        assert_non_negative(huber_delta, 'huber_delta')
        assert_positive(tol, 'tol')

        self.add_bias = add_bias
        self.interaction_penalty = interaction_penalty
        self.l1_penalty = l1_penalty
        self.huber_penalty = huber_penalty / huber_delta
        self.huber_delta = huber_delta
        self.tol = tol

        self._weights = None
        self._bias = None

        self._optimizer = self.__get_optimizer()

    def __get_optimizer(self):
        if self.l1_penalty > 0:
            optimizer = ProximalGradientDescentOptimizer(conv_threshold=self.tol)
            g, prox = l1_penalty_func_and_prox(self.l1_penalty, self.add_bias)
            return lambda x0, func: optimizer.minimize(x0, func.compute_loss, lambda x: func(x, return_matrix=True), g, prox)

        return lambda x0, func: scipy.optimize.minimize(func, x0, jac=True, method='bfgs', options={'gtol': self.tol})

    @property
    def bias(self):
        if self._bias is None:
            return None

        return self._bias.copy() if self.add_bias else np.zeros(self._weights.shape[0])

    @property
    def weights(self):
        return None if self._weights is None else self._weights.copy()

    @property
    def num_subspaces(self):
        return 0 if self._weights is None else self._weights.shape[0]

    def copy(self) -> LinearFactorizationLearner:
        learner = LinearFactorizationLearner(add_bias=self.add_bias, interaction_penalty=self.interaction_penalty,
                                             l1_penalty=self.l1_penalty, huber_penalty=self.huber_penalty, huber_delta=self.huber_delta,
                                             tol=self.tol)
        learner._weights = self.weights
        if self.add_bias:
            learner._bias = self.bias

        return learner

    def fit(self, X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], x0: Optional[np.ndarray] = None):
        if isinstance(factorization, int):
            assert_positive_integer(factorization, 'factorization')
            num_subspaces = factorization
            factorization = None
        else:
            num_subspaces = len(factorization)

        loss = LinearFactorizationLoss(X=X, y=y, add_bias=self.add_bias,
                                       interaction_penalty=self.interaction_penalty,
                                       huber_penalty=self.huber_penalty, huber_delta=self.huber_delta,
                                       factorization=factorization)

        if x0 is None:
            x0 = np.random.uniform(-1, 1, size=(num_subspaces, loss.X.shape[1]))

        if loss.factorization is not None:
            x0 = x0[loss.factorization]

        opt_result = self._optimizer(x0, loss)

        self._weights = self.__sort_matrix(loss.get_weights_matrix(opt_result.x))  # sort matrix in order to make weights more consistent
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
        margin = X @ self._weights.T
        if self.add_bias:
            margin += self._bias
        return margin

    @staticmethod
    def __sort_matrix(weights: np.ndarray) -> np.ndarray:
        return np.array(sorted(list(x) for x in weights))


class LinearFactorizationLoss:
    def __init__(self, X: np.ndarray, y: np.ndarray, add_bias: bool = True, interaction_penalty: float = 0,
                 huber_penalty: float = 0, huber_delta: float = 1e-3, factorization: Optional[List[List[int]]] = None):
        if add_bias:
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        if factorization is not None:
            B = np.full((len(factorization), X.shape[1]), False)
            for i, p in enumerate(factorization):
                B[i, p] = True
            if add_bias:
                B[:, -1] = True
            factorization = B

        self.X = X
        self.y = y
        self.add_bias = add_bias
        self.interaction_penalty = interaction_penalty
        self.huber_penalty = huber_penalty
        self.huber_delta = huber_delta
        self.factorization = factorization

    def __call__(self, weights: np.ndarray, return_matrix: bool = False):
        weights = self.get_weights_matrix(weights)

        margins = self.X @ weights.T
        loss, grad_weights = utils.compute_loss_and_grad(margins, self.y)
        grads = grad_weights.T @ self.X

        if self.interaction_penalty > 0 and weights.shape[0] > 1:
            loss += self.__add_penalty(self._compute_interaction_penalty_and_grad, weights, grads)

        if self.huber_penalty > 0:
            loss += self.__add_penalty(self._compute_huber_penalty_and_grad, weights, grads)

        if self.factorization is not None:
            if return_matrix:
                grads[~self.factorization] = 0  # only works for first-order optimizers (e.g. gradient descent)
            else:
                grads = grads[self.factorization]

        return loss, (grads if return_matrix else grads.ravel())

    def compute_loss(self, weights: np.ndarray):
        weights = self.get_weights_matrix(weights)

        margins = self.X @ weights.T
        loss = utils.compute_loss(margins, self.y)

        if self.interaction_penalty > 0 and weights.shape[0] > 1:
            loss += self._compute_interaction_penalty(weights)

        if self.huber_penalty > 0:
            loss += self._compute_huber_penalty(weights)

        return loss

    def get_weights_matrix(self, weights):
        if self.factorization is None:
            return weights.reshape(-1, self.X.shape[1])

        w = np.zeros((len(self.factorization), self.X.shape[1]))
        w[self.factorization] = weights
        return w

    def _compute_interaction_penalty_and_grad(self, weights):
        penalty, weights, wsq = self.__interaction_penalty_helper(weights)

        col_sq = np.sum(wsq, axis=0)
        grad = (2 * self.interaction_penalty) * weights * (col_sq - wsq)
        return penalty, grad

    def _compute_interaction_penalty(self, weights):
        return self.__interaction_penalty_helper(weights)[0]

    def __interaction_penalty_helper(self, weights):
        if self.add_bias:
            weights = weights[:, :-1]

        wsq = np.square(weights)

        M = wsq @ wsq.T
        np.fill_diagonal(M, 0)
        penalty = 0.5 * self.interaction_penalty * M.sum()
        return penalty, weights, wsq

    def _compute_huber_penalty_and_grad(self, weights):
        return utils.compute_huber_penalty_and_grad(weights, self.huber_penalty, self.huber_delta, int(self.add_bias))

    def _compute_huber_penalty(self, weights):
        return utils.compute_huber_penalty(weights, self.huber_penalty, self.huber_delta, int(self.add_bias))

    def __add_penalty(self, penalty_func, weights, grads):
        penalty_loss, grads_loss = penalty_func(weights)

        idx = grads.shape[1] - self.add_bias
        grads[:, :idx] += grads_loss

        return penalty_loss
