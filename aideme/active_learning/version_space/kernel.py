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

from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.linalg
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

if TYPE_CHECKING:
    from .linear import BayesianLogisticRegressionBase


class KernelBayesianLogisticRegression:
    """
    Add kernel support to LinearBayesianLogisticRegression classifier. Basically, the data matrix X is substituted by
    the Kernel matrix K, depending on the chosen kernel ('linear', 'rbf', 'poly', or user-defined).
    """

    def __init__(self, logreg: BayesianLogisticRegressionBase, decompose: bool = False,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        self.logreg = logreg
        self.decompose = decompose
        self.jitter = jitter
        self.kernel = self.__get_kernel(kernel, gamma, degree, coef0)

    @staticmethod
    def __get_kernel(kernel: str, gamma: Optional[float], degree: int, coef0: float):
        if kernel == 'linear':
            return linear_kernel
        elif kernel == 'poly':
            return partial(polynomial_kernel, gamma=gamma, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            return partial(rbf_kernel, gamma=gamma)
        elif callable(kernel):
            return kernel

        raise ValueError("Unsupported kernel. Available options are 'linear', 'rbf', 'poly', or any custom K(X,Y) function.")

    def __preprocess(self, X: np.ndarray) -> np.ndarray:
        return self.kernel(X, self.X_train)

    def clear(self):
        self.logreg.clear()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X

        K = self.__preprocess(X)

        if self.decompose:
            K[np.diag_indices_from(K)] += self.jitter
            K = np.linalg.cholesky(K)

        self.logreg.fit(K, y)

        if self.decompose:
            self.logreg.weight = scipy.linalg.solve_triangular(K, self.logreg.weight.T, lower=True, trans=1).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.logreg.predict(self.__preprocess(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.logreg.predict_proba(self.__preprocess(X))
