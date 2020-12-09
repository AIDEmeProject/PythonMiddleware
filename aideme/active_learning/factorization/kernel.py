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

from typing import Optional

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
        self.__X_train = None

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
        learner_copy.__X_train = self.__X_train
        return learner_copy

    def _get_loss(self, X: np.ndarray, y: np.ndarray):
        K = self._get_kernel_matrix(X, cholesky=True)
        return self.fact_linear._get_loss(K, y)

    def fit(self, X: np.ndarray, y: np.ndarray, factorization: int, x0: Optional[np.ndarray] = None):
        self.__X_train = X
        K = self._get_kernel_matrix(self.__X_train, cholesky=True)

        if x0 is not None:
            x0[:, :-1] = x0[:, :-1] @ K

        self.fact_linear.fit(K, y, factorization, x0)
        self.fact_linear._weights = scipy.linalg.solve_triangular(K, self.fact_linear._weights.T, lower=True, trans=1).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict(self._get_kernel_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict_proba(self._get_kernel_matrix(X))

    def partial_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.partial_proba(self._get_kernel_matrix(X))

    def _get_kernel_matrix(self, X: np.ndarray, cholesky: bool = False) -> np.ndarray:
        K = self.kernel(X, self.__X_train)
        if cholesky:
            scipy.linalg.cholesky(K.T, lower=False, overwrite_a=True)  # inplace Cholesky decomposition
        return K
