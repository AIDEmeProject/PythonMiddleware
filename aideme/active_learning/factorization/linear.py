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

from typing import List, Union, TYPE_CHECKING, Tuple, Optional

from scipy.special import expit

from aideme.utils import assert_positive_integer
from .optimization import ProximalGradientDescent
from .penalty import *

if TYPE_CHECKING:
    from .optimization import OptimizationAlgorithm


class LinearFactorizationLearner:
    def __init__(self, optimizer: OptimizationAlgorithm, penalty_term: Optional[PenaltyTerm] = None, add_bias: bool = True):
        self.optimizer = optimizer
        self.penalty_term = self.__process_proximal_penalty(penalty_term, optimizer)
        self.add_bias = add_bias
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

    def clear(self) -> None:
        self._weights = None
        self._bias = None

    @property
    def feature_groups(self) -> Optional[List[List[int]]]:
        return getattr(self.penalty_term, 'groups', None)

    @feature_groups.setter
    def feature_groups(self, value) -> None:
        if hasattr(self.penalty_term, 'groups'):
            self.penalty_term.groups = value

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
        learner = LinearFactorizationLearner(optimizer=self.optimizer, add_bias=self.add_bias)
        learner.penalty_term = self.penalty_term

        learner._weights = self.weights
        if self.add_bias:
            learner._bias = self.bias

        return learner

    def fit(self, X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], retries: int = 1, x0: Optional[np.ndarray] = None):
        assert_positive_integer(retries, 'retries')

        if isinstance(factorization, int):
            assert_positive_integer(factorization, 'factorization')
            num_subspaces = factorization
            factorization = None
        else:
            num_subspaces = len(factorization)

        loss = self._get_loss(X, y, factorization)
        if hasattr(self.optimizer, 'batch_size'):
            loss.set_batch_size(self.optimizer.batch_size)

        if x0 is None:
            x0 = np.random.normal(size=(retries, num_subspaces, loss.dim))
        else:
            x0 = x0.reshape((retries, num_subspaces, loss.dim))

        if factorization is not None:
            x0 = x0[:, loss.factorization]

        opt_result = self.__run_optimizer(loss, x0)
        self._bias, self._weights = loss.get_weights_matrix(opt_result.x)

        return opt_result

    def __run_optimizer(self, loss, x0):
        opt_result, min_val = None, np.inf

        for starting_point in x0:
            result = self.optimizer.minimize(starting_point, loss.compute_loss, loss.compute_grad)
            if result.fun < min_val or min_val == np.inf:
                opt_result, min_val = result, result.fun

        if min_val == np.inf:
            raise RuntimeError("Optimization failed:\n{}".format(opt_result))

        return opt_result

    def _get_loss(self, X: np.ndarray, y: np.ndarray, factorization: Optional[List[List[int]]] = None) -> LinearFactorizationLoss:
        return LinearFactorizationLoss(X=X, y=y, add_bias=self.add_bias, penalty_term=self.penalty_term, factorization=factorization)

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probas = utils.compute_log_probas(self._margin(X))
        return np.where(log_probas > np.log(0.5), 1., 0.)

    def partial_predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.partial_proba(X) > 0.5, 1., 0.)

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
    def __init__(self, X: np.ndarray, y: np.ndarray, add_bias: bool = True,
                 penalty_term: PenaltyTerm = None, factorization: Optional[List[List[int]]] = None):
        self.X = X
        self.y = y
        self.add_bias = add_bias
        self.penalty_term = penalty_term
        self.factorization = self.__compute_factorization_matrix(factorization)
        self._batch_size = None
        self._offsets = None
        self._cur_pos = None

    def __compute_factorization_matrix(self, factorization: Optional[List[List[int]]]) -> Optional[np.ndarray]:
        if factorization is None:
            return factorization

        B = np.full((len(factorization), self.dim), False)
        for i, p in enumerate(factorization):
            B[i, p] = True

        if self.add_bias:
            B[:, -1] = True

        return B

    @property
    def dim(self) -> int:
        return self.X.shape[1] + self.add_bias

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
        bias, weights = self.get_weights_matrix(weights)

        margins = self.__compute_margin(self.X, bias, weights)
        loss = utils.compute_loss(margins, self.y)

        # add penalty terms
        if self.penalty_term is not None:
            loss += self.penalty_term.loss(weights)

        return loss

    def compute_grad(self, weights: np.ndarray):
        bias, weights = self.get_weights_matrix(weights)

        X, y = self.X, self.y
        if self._batch_size is not None:
            idx = self.get_next_batch()
            X, y = self.X[idx], self.y[idx]

        margins = self.__compute_margin(X, bias, weights)
        grad_weights = self.__compute_grad_weights(margins, y)
        grad_b, grad_w = self.__compute_grad(X, grad_weights)

        # add penalty terms
        if self.penalty_term is not None:
            grad_w += self.penalty_term.grad(weights)

        # restrict weights to factorization selection
        grads = np.hstack([grad_w, grad_b.reshape(-1, 1)]) if self.add_bias else grad_w

        if self.factorization is not None:
            grads = grads[self.factorization]

        return grads

    def __compute_margin(self, X: np.ndarray, bias: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            margin = np.einsum('ijk, kj -> ik', X, weights)
        else:
            margin = X @ weights.T

        if self.add_bias:
            margin += bias

        return margin

    def __compute_grad(self, X: np.ndarray, grad_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 3:
            grad_w = np.einsum('ijk, ik -> kj', X, grad_weights)
        else:
            grad_w = grad_weights.T @ X

        grad_b = None
        if self.add_bias:
            grad_b = grad_weights.sum(axis=0)

        return grad_b, grad_w

    def __compute_grad_weights(self, margins: np.ndarray, y: np.ndarray):
        probas = expit(margins)

        weights = probas.prod(axis=1)
        np.true_divide(weights, 1 - weights, out=weights, where=weights < 1)
        weights[y > 0] = -1
        weights /= margins.shape[0]

        return (1 - probas) * weights.reshape(-1, 1)

    def get_weights_matrix(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.factorization is None:
            w = weights.reshape(-1, self.dim)
        else:
            w = np.zeros((len(self.factorization), self.dim))
            w[self.factorization] = weights

        b = None
        if self.add_bias:
            b, w = w[:, -1], w[:, :-1]

        return b, w
