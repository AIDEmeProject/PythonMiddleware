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

import numpy as np
import scipy

from ..kernel import Kernel, IncrementedDiagonalKernel


class KernelBayesianLogisticRegression:
    """
    Add kernel support to LinearBayesianLogisticRegression classifier. Basically, the data matrix X is substituted by
    the Kernel matrix K, depending on the chosen kernel ('linear', 'rbf', 'poly', or user-defined).
    """
    def __init__(self, logreg, decompose: bool = False, jitter: float = 1e-12,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0.):
        self.__logreg = logreg
        self.__decompose = decompose

        self.kernel = Kernel.get(kernel, gamma=gamma, degree=degree, coef0=coef0)
        if self.__decompose:
            self.kernel = IncrementedDiagonalKernel(self.kernel, jitter=jitter)

        self.__X_train = None
        self.__L_train = None

    def clear(self) -> None:
        self.__X_train = None
        self.__L_train = None
        self.__logreg.clear()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        K = self.kernel(X)

        if self.__decompose:
            scipy.linalg.cholesky(K.T, lower=False, overwrite_a=True)  # inplace Cholesky decomposition

            self.__L_train = K

            K = np.c_[K, np.zeros(K.shape[0])]

        self.__logreg.fit(K, y)

        self.__X_train = X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__logreg.predict(self.__get_kernel_matrix(X))

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        return self.__logreg._likelihood(self.__get_kernel_matrix(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.__logreg.predict_proba(self.__get_kernel_matrix(X))

    def __get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        K = self.kernel(X, self.__X_train)

        if self.__decompose:
            scipy.linalg.solve_triangular(self.__L_train, K.T, lower=True, trans=0, overwrite_b=True)  # inplace L^-1 K

            delta = self.kernel.diagonal(X) - np.einsum('ir, ir -> i', K, K)
            np.sqrt(delta, out=delta)

            K = np.c_[K, delta]  # TODO: how to avoid copying data?

        return K
