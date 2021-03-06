#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.
from __future__ import annotations

from typing import Optional, List

import numpy as np
from sklearn.base import clone
from sklearn.kernel_approximation import Nystroem
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
    def __init__(self, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1):
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

        if Y is None or Y is X:
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


class KernelTransformer:
    @staticmethod
    def get(kernel: str = 'rbf', degree: int = 3, gamma: Optional[float] = None, coef0: float = 1, jitter: float = 0,
            nystroem_components: Optional[int] = None) -> KernelTransformer:
        if nystroem_components is not None:
            return NystroemTransformer(n_components=nystroem_components, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)

        kernel = Kernel.get(kernel, degree=degree, gamma=gamma, coef0=coef0)
        if jitter > 0:
            kernel = IncrementedDiagonalKernel(kernel, jitter)

        return KernelTransformerWrapper(kernel)

    @property
    def n_components(self) -> int:
        raise NotImplementedError

    def clone(self) -> KernelTransformer:
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class KernelTransformerWrapper(KernelTransformer):
    def __init__(self, kernel: Kernel):
        self._kernel = kernel
        self._X_train = None

    @property
    def n_components(self) -> int:
        return self._X_train.shape[0]

    def clone(self) -> KernelTransformer:
        return KernelTransformerWrapper(self._kernel)

    def fit(self, X: np.ndarray) -> None:
        self._X_train = X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._kernel(X, self._X_train)


class NystroemTransformer(KernelTransformer):
    def __init__(self, n_components: int = 100, kernel: str = 'rbf', degree: int = 3, gamma: Optional[float] = None, coef0: float = 1):
        self.nystroem = Nystroem(n_components=n_components, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)

    @property
    def n_components(self) -> int:
        return self.nystroem.n_components

    def clone(self) -> KernelTransformer:
        kt = NystroemTransformer()
        kt.nystroem = clone(self.nystroem)
        return kt

    def fit(self, X: np.ndarray) -> None:
        self.nystroem.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.nystroem.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.nystroem.fit_transform(X)


class FactorizedKernelTransform(KernelTransformer):
    def __init__(self, base_transformer: KernelTransformer, factorization: List[List[int]]):
        self._base = base_transformer
        self._factorization = factorization
        self._transformers = [base_transformer.clone() for _ in factorization]

    @property
    def n_components(self) -> int:
        return self._transformers[0].n_components

    def clone(self) -> KernelTransformer:
        return FactorizedKernelTransform(self._base, self._factorization)

    def fit(self, X: np.ndarray) -> None:
        for transformer, subspace in zip(self._transformers, self._factorization):
            transformer.fit(X[:, subspace])

    def transform(self, X: np.ndarray) -> np.ndarray:
        K = np.empty((X.shape[0], self.n_components, len(self._factorization)))
        for k, (transformer, subspace) in enumerate(zip(self._transformers, self._factorization)):
            K[:, :, k] = transformer.transform(X[:, subspace])
        return K

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        K = np.empty((X.shape[0], self.n_components, len(self._factorization)))
        for k, (transformer, subspace) in enumerate(zip(self._transformers, self._factorization)):
            K[:, :, k] = transformer.fit_transform(X[:, subspace])
        return K
