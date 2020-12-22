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

from typing import Optional, Union, List

import numpy as np
import scipy.linalg

from .linear import LinearFactorizationLearner
from .optimization import OptimizationAlgorithm
from ..kernel import Kernel, IncrementedDiagonalKernel


class KernelFactorizationLearner:
    def __init__(self, optimizer: OptimizationAlgorithm, add_bias: bool = True, interaction_penalty: float = 0,
                 l1_penalty: float = 0,  l2_penalty: float = 0, l2_sqrt_penalty: float = 0, l2_sqrt_weights: Optional[np.ndarray] = None,
                 huber_penalty: float = 0, huber_delta: float = 1e-3,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):

        self.fact_linear = LinearFactorizationLearner(
            optimizer=optimizer, add_bias=add_bias, interaction_penalty=interaction_penalty,
            l1_penalty=l1_penalty,  l2_penalty=l2_penalty, l2_sqrt_penalty=l2_sqrt_penalty, l2_sqrt_weights=l2_sqrt_weights,
            huber_penalty=huber_penalty, huber_delta=huber_delta
        )
        self.kernel = Kernel.get(kernel, gamma=gamma, degree=degree, coef0=coef0)
        self.kernel = IncrementedDiagonalKernel(self.kernel, jitter=jitter)
        self._X_train = None
        self._factorization = None
        self._use_cholesky = len(self.fact_linear.penalty_terms) > 0

    @property
    def bias(self) -> np.ndarray:
        return self.fact_linear.bias

    @property
    def weights(self) -> np.ndarray:
        return self.fact_linear.weights

    @property
    def weight_matrix(self) -> Optional[np.ndarray]:
        return self.fact_linear.weight_matrix

    @property
    def num_subspaces(self) -> int:
        return self.fact_linear.num_subspaces

    def copy(self) -> KernelFactorizationLearner:
        learner_copy = KernelFactorizationLearner(self.fact_linear._optimizer)
        learner_copy.fact_linear = self.fact_linear.copy()
        learner_copy.kernel = self.kernel  # stateless
        learner_copy._X_train = self._X_train
        learner_copy._factorization = self._factorization
        return learner_copy

    def _get_loss(self, X: np.ndarray, y: np.ndarray):
        K = self._get_kernel_matrix(X, cholesky=self._use_cholesky)
        return self.fact_linear._get_loss(K, y)

    def fit(self, X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], retries: int = 1, x0: Optional[np.ndarray] = None):
        self._X_train = X

        if isinstance(factorization, int):
            self._factorization = None
        else:
            self._factorization = factorization
            factorization = len(factorization)

        K = self._get_kernel_matrix(self._X_train, cholesky=self._use_cholesky)

        if x0 is not None:
            if self.fact_linear.add_bias:
                x0[:, :-1] = x0[:, :-1] @ K
            else:
                x0 = x0 @ K

        self.fact_linear.fit(K, y, factorization, retries, x0)

        if self._use_cholesky:
            if self._factorization is None:
                self.fact_linear._weights = scipy.linalg.solve_triangular(K, self.fact_linear._weights.T, lower=True, trans=1).T
            else:
                for k in range(K.shape[2]):
                    self.fact_linear._weights[k] = scipy.linalg.solve_triangular(K[:, :, k], self.fact_linear._weights[k], lower=True, trans=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict(self._get_kernel_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict_proba(self._get_kernel_matrix(X))

    def partial_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.partial_proba(self._get_kernel_matrix(X))

    def _get_kernel_matrix(self, X: np.ndarray, cholesky: bool = False) -> np.ndarray:
        if self._factorization is None:
            return self.__kernel_matrix_helper(X, cholesky)

        return np.concatenate([self.__kernel_matrix_helper(X, cholesky, p)[:, :, np.newaxis] for p in self._factorization], axis=2)

    def __kernel_matrix_helper(self, X: np.ndarray, cholesky: bool, subspace: Optional[List[int]] = None) -> np.ndarray:
        X1, X2 = X, self._X_train
        if subspace is not None:
            X1 = X[:, subspace]
            X2 = X1 if X is self._X_train else self._X_train[:, subspace]

        K = self.kernel(X1, X2)
        if cholesky:
            scipy.linalg.cholesky(K.T, lower=False, overwrite_a=True)  # inplace Cholesky decomposition
        return K
