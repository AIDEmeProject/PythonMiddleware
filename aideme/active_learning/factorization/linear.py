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

from typing import List, Union, TYPE_CHECKING

from scipy.special import expit

from aideme.utils import assert_positive_integer
from .optimization import ProximalGradientDescent
from .penalty import *

if TYPE_CHECKING:
    from .optimization import OptimizationAlgorithm

import memory_profiler

class LinearFactorizationLearner:
    def __init__(self, optimizer: OptimizationAlgorithm, add_bias: bool = True, interaction_penalty: float = 0,
                 l1_penalty: float = 0,  l2_penalty: float = 0, l2_sqrt_penalty: float = 0, l2_sqrt_weights: Optional[np.ndarray] = None,
                 huber_penalty: float = 0, huber_delta: float = 1e-3):
        self._optimizer = optimizer
        self.add_bias = add_bias

        self.penalty_terms = []
        if l1_penalty > 0 and l2_sqrt_penalty > 0:
            self.penalty_terms.append(self.__process_proximal_penalty(SparseGroupLassoPenalty(l1_penalty, l2_sqrt_penalty), optimizer))
        elif l1_penalty > 0:
            self.penalty_terms.append(self.__process_proximal_penalty(L1Penalty(l1_penalty), optimizer))
        elif l2_sqrt_penalty > 0:
            self.penalty_terms.append(self.__process_proximal_penalty(L2SqrtPenalty(l2_sqrt_penalty, l2_sqrt_weights), optimizer))

        if l2_penalty > 0:
            self.penalty_terms.append(L2Penalty(l2_penalty))
        if interaction_penalty > 0:
            self.penalty_terms.append(InteractionPenalty(interaction_penalty))
        if huber_penalty > 0:
            self.penalty_terms.append(HuberPenalty(huber_penalty, huber_delta))

        self._weights = None
        self._bias = None

    def __process_proximal_penalty(self, penalty_term: PenaltyTerm, optimizer: OptimizationAlgorithm) -> PenaltyTerm:
        if isinstance(optimizer, ProximalGradientDescent):
            # In the special case of ProximalGradientDescent algorithm, we must be careful to not mistakenly add
            # the penalty gradient to the loss function gradient. Additionally, we must be careful to exclude the bias
            # column from proximal function computations
            penalty_term.grad = lambda x: 0
            optimizer.remove_bias_column = self.add_bias
            optimizer.penalty_term = penalty_term

        return penalty_term

    @property
    def bias(self) -> Optional[np.ndarray]:
        if self._bias is None:
            return None

        return self._bias.copy() if self.add_bias else np.zeros(self._weights.shape[0])

    @property
    def weights(self) -> Optional[np.ndarray]:
        return None if self._weights is None else self._weights.copy()

    @property
    def weight_matrix(self) -> Optional[np.ndarray]:
        if self._weights is None:
            return None

        return np.hstack([self._weights, self._bias.reshape(-1, 1)]) if self.add_bias else self.weights

    @property
    def num_subspaces(self) -> int:
        return 0 if self._weights is None else self._weights.shape[0]

    def copy(self) -> LinearFactorizationLearner:
        learner = LinearFactorizationLearner(optimizer=self._optimizer, add_bias=self.add_bias)
        learner.penalty_terms = self.penalty_terms

        learner._weights = self.weights
        if self.add_bias:
            learner._bias = self.bias

        return learner

    @memory_profiler.profile
    def fit(self, X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], retries: int = 1, x0: Optional[np.ndarray] = None):
        assert_positive_integer(retries, 'retries')

        if isinstance(factorization, int):
            assert_positive_integer(factorization, 'factorization')
            num_subspaces = factorization
            factorization = None
        else:
            num_subspaces = len(factorization)

        loss = self._get_loss(X, y, factorization)
        if hasattr(self._optimizer, 'batch_size'):
            loss.set_batch_size(self._optimizer.batch_size)

        if x0 is None:
            x0 = np.random.normal(size=(retries, num_subspaces, loss.X.shape[1]))
        else:
            x0 = x0.reshape((retries, num_subspaces, loss.X.shape[1]))

        if factorization is not None:
            x0 = x0[:, loss.factorization]

        opt_result, min_val = None, np.inf
        for starting_point in x0:
            result = self._optimizer.minimize(starting_point, loss.compute_loss, loss.compute_grad)
            if result.fun < min_val:
                opt_result, min_val = result, result.fun

        self._weights = loss.get_weights_matrix(opt_result.x)  # sort matrix in order to make weights more consistent
        if self.add_bias:
            self._bias = self._weights[:, -1]
            self._weights = self._weights[:, :-1]

        return opt_result

    def _get_loss(self, X: np.ndarray, y: np.ndarray, factorization: Optional[List[List[int]]] = None) -> LinearFactorizationLoss:
        return LinearFactorizationLoss(X=X, y=y, add_bias=self.add_bias, penalty_terms=self.penalty_terms, factorization=factorization)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probas = utils.compute_log_probas(self._margin(X))
        return np.where(log_probas > np.log(0.5), 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probas = utils.compute_log_probas(self._margin(X))
        return np.exp(log_probas)

    def partial_proba(self, X: np.ndarray) -> np.ndarray:
        return expit(self._margin(X))

    def _margin(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            margin = np.einsum('ijk, kj -> ik', X, self._weights)
        else:
            margin = X @ self._weights.T

        if self.add_bias:
            margin += self._bias
        return margin


class LinearFactorizationLoss:
    @memory_profiler.profile
    def __init__(self, X: np.ndarray, y: np.ndarray, add_bias: bool = True,
                 penalty_terms: List[PenaltyTerm] = None, factorization: Optional[List[List[int]]] = None):
        if add_bias:
            X = self.__add_bias_column(X)

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

        self._batch_size = None
        self._offsets = None
        self._cur_pos = None

    @classmethod
    def __add_bias_column(cls, X):
        if X.ndim == 2:
            return cls.__add_bias_column_helper(X)

        X_with_bias = np.empty((X.shape[0], X.shape[1] + 1, X.shape[2]))
        for k in range(X.shape[2]):
            X_with_bias[:, :, k] = cls.__add_bias_column_helper(X[:, :, k])
        return X_with_bias

    @staticmethod
    def __add_bias_column_helper(X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def set_batch_size(self, batch_size: Optional[int]):
        if batch_size is None or batch_size >= len(self.X):
            return

        assert_positive_integer(batch_size, 'batch_size')
        self._batch_size = batch_size
        self._offsets = np.arange(len(self.X), dtype='int')
        np.random.shuffle(self._offsets)
        self._cur_pos = 0

    def get_next_batch(self):
        if self._cur_pos >= len(self.X):
            self._cur_pos = 0
            np.random.shuffle(self._offsets)

        end_pos = self._cur_pos + self._batch_size
        next_batch = self._offsets[self._cur_pos:end_pos]
        self._cur_pos = end_pos

        return next_batch

    def compute_loss(self, weights: np.ndarray):
        weights = self.get_weights_matrix(weights)

        margins = self.__compute_margin(self.X, weights)
        loss = utils.compute_loss(margins, self.y)

        # add penalty terms
        weights_wo_bias = self.__remove_bias(weights)
        for penalty in self.penalty_terms:
            loss += penalty.loss(weights_wo_bias)

        return loss

    def compute_grad(self, weights: np.ndarray):
        weights = self.get_weights_matrix(weights)

        X, y = self.X, self.y
        if self._batch_size is not None:
            idx = self.get_next_batch()
            X, y = self.X[idx], self.y[idx]

        margins = self.__compute_margin(X, weights)
        grad_weights = utils.compute_grad_factors(margins, y)
        grads = self.__compute_grad(X, grad_weights)

        # add penalty terms
        weights_wo_bias = self.__remove_bias(weights)
        grads_wo_bias = self.__remove_bias(grads)
        for penalty in self.penalty_terms:
            grads_wo_bias += penalty.grad(weights_wo_bias)

        # restrict weights to factorization selection
        if self.factorization is not None:
            grads = grads[self.factorization]

        return grads

    @staticmethod
    def __compute_margin(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return np.einsum('ijk, kj -> ik', X, weights)

        return X @ weights.T

    @staticmethod
    def __compute_grad(X: np.ndarray, grad_weights: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return np.einsum('ijk, ik -> kj', X, grad_weights)

        return grad_weights.T @ X

    def get_weights_matrix(self, weights):
        if self.factorization is None:
            return weights.reshape(-1, self.X.shape[1])

        w = np.zeros((len(self.factorization), self.X.shape[1]))
        w[self.factorization] = weights
        return w

    def __remove_bias(self, x: np.ndarray):
        return x[:, :-1] if self.add_bias else x
