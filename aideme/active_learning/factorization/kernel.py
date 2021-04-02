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

from typing import Optional, Union, List, TYPE_CHECKING

import numpy as np

from .linear import LinearFactorizationLearner
from .optimization import OptimizationAlgorithm
from ..kernel import KernelTransformer, FactorizedKernelTransform

if TYPE_CHECKING:
    from .penalty import PenaltyTerm


class KernelFactorizationLearner:
    def __init__(self, optimizer: OptimizationAlgorithm, penalty_term: PenaltyTerm, add_bias: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12, nystroem_components: Optional[int] = None):
        self.fact_linear = LinearFactorizationLearner(optimizer=optimizer, penalty_term=penalty_term, add_bias=add_bias)
        self._base_kernel_transform = KernelTransformer.get(kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter, nystroem_components=nystroem_components)
        self._kernel_transformer = None
        self._factorization = None

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

    def _get_loss(self, X: np.ndarray, y: np.ndarray):
        return self.fact_linear._get_loss(self._get_kernel_matrix(X), y)

    def fit(self, X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], retries: int = 1, x0: Optional[np.ndarray] = None):
        if isinstance(factorization, int):
            self._factorization = None
        else:
            self._factorization = factorization
            factorization = len(factorization)

        self._kernel_transformer = self._get_kernel_transform()
        K = self._kernel_transformer.fit_transform(X)

        return self.fact_linear.fit(K, y, factorization, retries, x0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict(self._get_kernel_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.predict_proba(self._get_kernel_matrix(X))

    def partial_proba(self, X: np.ndarray) -> np.ndarray:
        return self.fact_linear.partial_proba(self._get_kernel_matrix(X))

    def _get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        return self._kernel_transformer.transform(X)

    def _get_kernel_transform(self) -> KernelTransformer:
        if self._factorization is None:
            return self._base_kernel_transform.clone()

        return FactorizedKernelTransform(self._base_kernel_transform, self._factorization)
