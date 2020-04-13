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
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

from aideme.utils.validation import assert_positive, assert_positive_integer


class Kernel:
    @staticmethod
    def get(kernel: str, **kernel_params) -> Kernel:
        kernel = kernel.lower()

        if kernel == 'linear':
            return LinearKernel()

        if kernel == 'rbf' or kernel == 'gaussian':
            return GaussianKernel(
                gamma=kernel_params.get('gamma', None)
            )

        if kernel == 'poly':
            return PolynomialKernel(
                degree=kernel_params.get('degree', 3),
                gamma=kernel_params.get('gamma', None),
                coef0=kernel_params.get('coef0', 1),
            )

        raise ValueError("Unknown kernel function {}. Possible values are: 'linear', 'rbf' / 'gaussian', and 'poly'.")

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the kernel matrix of all pairs of rows (X_i, Y_j). Simply calls the compute() method.

        :param X: a M x D matrix of data points
        :param Y: a N x D matrix of data points. If None, we set Y = X
        :return: the M x N kernel matrix K_ij = k(X_i, Y_j)
        """
        return self.compute(X, Y)

    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the kernel matrix of all pairs of rows (X_i, Y_j).

        :param X: a M x D matrix of data points
        :param Y: a N x D matrix of data points. If None, we set Y = X
        :return: the M x N kernel matrix K_ij = k(X_i, Y_j)
        """
        raise NotImplementedError

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the diagonal of the kernel matrix of the data matrix X

        :param X: a M x D matrix of data points
        :return: a M-dimensional array K_i = k(X_i, X_i)
        """
        raise NotImplementedError


class LinearKernel(Kernel):
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the kernel matrix of all pairs of rows (X_i, Y_j).

        :param X: a M x D matrix of data points
        :param Y: a N x D matrix of data points. If None, we set Y = X
        :return: the M x N kernel matrix K_ij = k(X_i, Y_j)
        """
        return linear_kernel(X, Y)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the diagonal of the kernel matrix of the data matrix X

        :param X: a M x D matrix of data points
        :return: a M-dimensional array K_i = k(X_i, X_i)
        """
        return np.einsum('ir, ir -> i', X, X)


class GaussianKernel(Kernel):
    def __init__(self, gamma: Optional[float] = None):
        """
        :param gamma: gaussian kernel parameter. If None, we use 1 / n_features
        """
        assert_positive(gamma, 'gamma', allow_none=True)
        self.gamma = gamma

    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the kernel matrix of all pairs of rows (X_i, Y_j).

        :param X: a M x D matrix of data points
        :param Y: a N x D matrix of data points. If None, we set Y = X
        :return: the M x N kernel matrix K_ij = k(X_i, Y_j)
        """
        return rbf_kernel(X, Y, gamma=self.gamma)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the diagonal of the kernel matrix of the data matrix X

        :param X: a M x D matrix of data points
        :return: a M-dimensional array K_i = k(X_i, X_i)
        """
        return np.ones(len(X), dtype=np.float)


class PolynomialKernel(Kernel):
    def __init__(self, degree: int = 3, gamma: Optional[int] = None, coef0: float = 1):
        """
        (gamma <X, Y> + coef0)^degree
        :param gamma: gaussian kernel parameter. If None, we use 1 / n_features
        :param
        """
        assert_positive_integer(degree, 'degree')
        assert_positive(gamma, 'gamma', allow_none=True)

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the kernel matrix of all pairs of rows (X_i, Y_j).

        :param X: a M x D matrix of data points
        :param Y: a N x D matrix of data points. If None, we set Y = X
        :return: the M x N kernel matrix K_ij = k(X_i, Y_j)
        """
        return polynomial_kernel(X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the diagonal of the kernel matrix of the data matrix X

        :param X: a M x D matrix of data points
        :return: a M-dimensional array K_i = k(X_i, X_i)
        """
        sqsum = np.einsum('ir, ir -> i', X, X)
        return (self.gamma * sqsum + self.coef0) ** (1 / self.degree)


class IncrementedDiagonalKernel(Kernel):
    """
    Given a kernel k(x, y), it computes 'incremented kernel':

            k_inc(x, y) = k(x, y) + lambda * 1(x == y)

    i.e., it adds a factor of lambda whenever x and y are identical.

    This transformation is useful when we need a positive-definite kernel matrix K.
    """
    def __init__(self, kernel: Kernel, jitter: float):
        """
        :param kernel: kernel object
        :param jitter: positive value to add to diagonal
        """
        assert_positive(jitter, 'jitter')

        self._kernel = kernel
        self._jitter = jitter

    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the kernel matrix of all pairs of rows (X_i, Y_j). When Y is None, we add the jitter to the diagonal

        :param X: a M x D matrix of data points
        :param Y: a N x D matrix of data points. If None, we set Y = X
        :return: the M x N kernel matrix K_ij = k(X_i, Y_j)
        """
        K = self._kernel(X, Y)

        if Y is None:
            K[np.diag_indices_from(K)] += self._jitter

        return K

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the diagonal of the kernel matrix computed over the data X

        :param X: a M x D matrix of data points
        :return: a M-dimensional array K_i = k(X_i, X_i) + jitter
        """
        diag = self._kernel.diagonal(X)
        diag += self._jitter
        return diag
