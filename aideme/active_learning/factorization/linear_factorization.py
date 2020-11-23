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

from typing import Optional, List, Union, TYPE_CHECKING

import numpy as np

import aideme.active_learning.factorization.utils as utils
from aideme.utils import assert_positive_integer
from .penalty import *

if TYPE_CHECKING:
    from .optimization import OptimizationAlgorithm


class LinearFactorizationLearner:
    def __init__(self, optimizer: OptimizationAlgorithm, add_bias: bool = True, interaction_penalty: float = 0,
                 l2_penalty: float = 0, huber_penalty: float = 0, huber_delta: float = 1e-3):
        self._optimizer = optimizer
        self.add_bias = add_bias

        self.penalty_terms = []
        if l2_penalty > 0:
            self.penalty_terms.append(L2Penalty(l2_penalty))
        if interaction_penalty > 0:
            self.penalty_terms.append(InteractionPenalty(interaction_penalty))
        if huber_penalty > 0:
            self.penalty_terms.append(HuberPenalty(huber_penalty, huber_delta))

        self._weights = None
        self._bias = None

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
        learner = LinearFactorizationLearner(optimizer=self._optimizer, add_bias=self.add_bias)
        learner.penalty_terms = self.penalty_terms

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

        loss = self._get_loss(X, y, factorization)

        if x0 is None:
            x0 = np.random.uniform(-1, 1, size=(num_subspaces, loss.X.shape[1]))

        if loss.factorization is not None:
            x0 = x0[loss.factorization]

        opt_result = self._optimizer.minimize(x0, loss.compute_loss, loss.compute_grad)

        self._weights = self.__sort_matrix(loss.get_weights_matrix(opt_result.x))  # sort matrix in order to make weights more consistent
        if self.add_bias:
            self._bias = self._weights[:, -1]
            self._weights = self._weights[:, :-1]

        return opt_result

    def _get_loss(self, X: np.ndarray, y: np.ndarray, factorization: Optional[List[List[int]]]) -> LinearFactorizationLoss:
        return LinearFactorizationLoss(X=X, y=y, add_bias=self.add_bias, penalty_terms=self.penalty_terms, factorization=factorization)

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
    def __init__(self, X: np.ndarray, y: np.ndarray, add_bias: bool = True,
                 penalty_terms: List[PenaltyTerm] = None, factorization: Optional[List[List[int]]] = None):
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
        self.penalty_terms = penalty_terms if penalty_terms is not None else []
        self.factorization = factorization

    def compute_loss(self, weights: np.ndarray):
        weights = self.get_weights_matrix(weights)

        margins = self.X @ weights.T
        loss = utils.compute_loss(margins, self.y)

        # add penalty terms
        weights_wo_bias = self.__remove_bias(weights)
        for penalty in self.penalty_terms:
            loss += penalty.loss(weights_wo_bias)

        return loss

    def compute_grad(self, weights: np.ndarray):
        weights = self.get_weights_matrix(weights)

        margins = self.X @ weights.T
        grad_weights = utils.compute_grad_factors(margins, self.y)
        grads = grad_weights.T @ self.X

        # add penalty terms
        weights_wo_bias = self.__remove_bias(weights)
        grads_wo_bias = self.__remove_bias(grads)
        for penalty in self.penalty_terms:
            grads_wo_bias += penalty.grad(weights_wo_bias)

        # restrict weights to factorization selection
        if self.factorization is not None:
            grads = grads[self.factorization]

        return grads

    def get_weights_matrix(self, weights):
        if self.factorization is None:
            return weights.reshape(-1, self.X.shape[1])

        w = np.zeros((len(self.factorization), self.X.shape[1]))
        w[self.factorization] = weights
        return w

    def __remove_bias(self, x: np.ndarray):
        return x[:, :-1] if self.add_bias else x
