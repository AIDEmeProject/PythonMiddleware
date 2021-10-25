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

from typing import Generator, List, Optional, Tuple
import warnings

import numpy as np
from scipy.special import expit
import scipy.optimize

import aideme.active_learning.factorization.utils as utils


class FactorizedLinearClassifier:
    __LOGHALF = np.log(0.5)

    def __init__(self, partition: List[List[int]], weights: Optional[np.ndarray] = None):
        self.partition = partition
        self._dim = len(partition) + sum(map(len, partition))

        if weights is None:
            weights = np.zeros(self._dim)

        if len(weights) != self._dim:
            raise ValueError("Incompatible dimension for weights: expected {}, but got {}".format(self._dim, len(weights)))

        self.weights = weights
        self.__offsets = np.cumsum([len(p) + 1 for p in self.partition])

    @property
    def dim(self) -> int:
        return self._dim

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self._log_proba(X) > self.__LOGHALF, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._log_proba(X))

    def partial_proba(self, X: np.ndarray, i: int) -> np.ndarray:
        begin = self.__offsets[i-1] if i > 0 else 0
        end = self.__offsets[i]
        return self._logistic_proba(X, self.weights[begin:end])

    def merge_partitions(self, i: int, j: int, merge_weights: bool = True) -> FactorizedLinearClassifier:
        if i == j:
            return self

        i, j = min(i, j), max(i, j)

        merged_part = (
                self.partition[:i]
                + self.partition[i + 1:j]
                + self.partition[j + 1:]
                + [self.partition[i] + self.partition[j]]
        )

        if not merge_weights:
            return FactorizedLinearClassifier(merged_part)

        begin_i, end_i = self.__offsets[i - 1] if i > 0 else 0, self.__offsets[i]
        begin_j, end_j = self.__offsets[j - 1], self.__offsets[j]

        merged_w = np.hstack([
            self.weights[:begin_i],
            self.weights[end_i:begin_j],
            self.weights[end_j:],
            [self.weights[begin_i] + self.weights[begin_j]],
            self.weights[begin_i + 1:end_i],
            self.weights[begin_j + 1:end_j]
        ])

        return FactorizedLinearClassifier(merged_part, merged_w)

    def _log_proba(self, X: np.ndarray) -> np.ndarray:
        log_probas = np.zeros(len(X), dtype='float')

        for X_partial, w_partial in self.__generate_subspace_data(X):
            margin = self._margin(X_partial, w_partial)
            log_probas += utils.log_sigmoid(margin)

        return log_probas

    def _grad_log_proba(self, X: np.ndarray) -> np.ndarray:
        grads = np.ones((self.dim, len(X)))

        for begin, end, X_partial, w_partial in self.__generate_subspace_data(X, include_endpoints=True):
            partial_proba = self._logistic_proba(X_partial, w_partial)

            grads[begin+1:end] = X_partial.T
            grads[begin:end] *= (1 - partial_proba)

        return grads

    def __generate_subspace_data(self, X: np.ndarray, include_endpoints: bool = False) -> Generator:
        begin = 0

        for p, end in zip(self.partition, self.__offsets):
            if include_endpoints:
                yield begin, end, X[:, p], self.weights[begin:end]
            else:
                yield X[:, p], self.weights[begin:end]
            begin = end

    @classmethod
    def _logistic_proba(cls, X, w):
        return expit(cls._margin(X, w))

    @staticmethod
    def _margin(X, w):
        return w[0] + X.dot(w[1:])


class FactorizedLinearLearner:
    def __init__(self, optimizer: str = 'BFGS', gtol: float = 1e-4, maxiter: Optional[int] = None):
        """
        :param optimizer: Scipy optimization method. Recommended options are 'BFGS' or 'CG'
        :param gtol: gradient threshold tolerance
        :param maxiter: maximum number of iterations. If None, no upper limit is set.
        """
        self._optimizer = optimizer
        self._opt_options = {'gtol': gtol, 'maxiter': maxiter}

    def fit(self, X: np.ndarray, y: np.ndarray, partition: List[List[int]], x0: Optional[np.ndarray] = None):
        return self.fit_and_loss(X, y, partition, x0)[0]

    def compute_factorization_loss(self, X: np.ndarray, y: np.ndarray, partition: List[List[int]], x0: Optional[np.ndarray] = None):
        return self.fit_and_loss(X, y, partition, x0)[1]

    def fit_and_loss(self, X: np.ndarray, y: np.ndarray, partition: List[List[int]], x0: Optional[np.ndarray] = None) -> Tuple[FactorizedLinearClassifier, float]:
        fact_clf = FactorizedLinearClassifier(partition, x0)
        loss = LinearLoss(X, y, fact_clf)

        res = scipy.optimize.minimize(loss, loss.x0, jac=True, method=self._optimizer, options=self._opt_options)

        if not res.success:
            warnings.warn("Optimization routine did not converge.\n{}".format(res))

        fact_clf.weights = res.x

        return fact_clf, res.fun


class LinearLoss:
    def __init__(self, X: np.ndarray, y: np.ndarray, fact_clf: FactorizedLinearClassifier):
        self.X = X
        self.y = y
        self.fact_clf = fact_clf
        self.x0 = self.fact_clf.weights.copy()

    def __call__(self, weights: np.ndarray) -> Tuple[float, np.ndarray]:
        self.fact_clf.weights = weights

        log_probas = self.fact_clf._log_proba(self.X)
        loss = utils.compute_classification_loss(log_probas, self.y)

        grads = self.fact_clf._grad_log_proba(self.X)
        weights = utils.grad_weights(log_probas, self.y)
        return loss, grads.dot(weights) / len(self.X)
