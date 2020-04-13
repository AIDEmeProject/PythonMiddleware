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

import numpy as np
import scipy

from aideme.utils import metric_logger
from ..kernel import Kernel


class KernelBayesianLogisticRegression:
    """
    Add kernel support to LinearBayesianLogisticRegression classifier. Basically, the data matrix X is substituted by
    the Kernel matrix K, depending on the chosen kernel ('linear', 'rbf', 'poly', or user-defined).
    """

    def __init__(self, logreg, decompose: bool = False,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        self.logreg = logreg
        self.decompose = decompose
        self.jitter = jitter
        self.kernel = Kernel.get(kernel, gamma=gamma, degree=degree, coef0=coef0)
        self.X_train = None
        self.L_train = None

    def clear(self) -> None:
        self.X_train = None
        self.L_train = None
        self.logreg.clear()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        K = self.__preprocess_fit(X)

        self.X_train = X
        if self.decompose:
            self.L_train = K
            K = np.hstack([K, np.zeros(shape=(len(K), 1))])

        self.logreg.fit(K, y)

    @metric_logger.log_execution_time('preprocess_fit')
    def __preprocess_fit(self, X_train):
        K = self.kernel(X_train)

        if not self.decompose:
            return K

        # inplace Cholesky decomposition
        K[np.diag_indices_from(K)] += self.jitter
        scipy.linalg.cholesky(K.T, lower=False, overwrite_a=True)  # inplace Cholesky decomposition

        return K

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.logreg.predict_proba(self.__preprocess_predict(X))

    @metric_logger.log_execution_time('preprocess_predict', 'append')
    def __preprocess_predict(self, X: np.ndarray) -> np.ndarray:
        K = self.kernel(X, self.X_train)

        if not self.decompose:
            return K

        # solve the system L^-1 K inplace
        scipy.linalg.solve_triangular(self.L_train, K.T, lower=True, trans=0, overwrite_b=True)

        sqnorm = np.einsum('ir, ir -> i', K, K).reshape(-1, 1)
        K = np.hstack([K, np.sqrt(self.kernel.diagonal(X) - sqnorm)])

        return K
