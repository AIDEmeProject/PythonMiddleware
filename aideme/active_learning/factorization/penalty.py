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

import aideme.active_learning.factorization.utils as utils
from aideme.utils import assert_positive


class PenaltyTerm:
    def __init__(self, penalty: float):
        self.penalty = penalty

    @property
    def penalty(self) -> float:
        return self._penalty

    @penalty.setter
    def penalty(self, value) -> None:
        assert_positive(value, 'penalty')
        self._penalty = value

    def loss(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        """
        :return: argmin_y pen(y) + || x - y ||^2 / 2 * eta
        """
        return None

    def is_subgradient(self, v: np.ndarray, x: np.ndarray, tol: float) -> bool:
        return np.linalg.norm(v - self.grad(x)) < tol


class L1Penalty(PenaltyTerm):
    def loss(self, x: np.ndarray) -> float:
        return self._penalty * np.abs(x).sum()

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._penalty * np.sign(x)

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - self._penalty * eta, 0)

    def is_subgradient(self, v: np.ndarray, x: np.ndarray, tol: float) -> bool:
        sign = self._penalty * np.sign(x)
        u = np.abs(v - sign)
        u[sign == 0] -= self._penalty
        return np.all(u <= tol)


class L2SqrtPenalty(PenaltyTerm):
    def __init__(self, penalty: float, groups = None):
        super().__init__(penalty)
        self.groups = groups

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, value) -> None:
        if value is None:
            value = [slice(None)]
        self._groups = value

    def loss(self, x: np.ndarray) -> float:
        s = 0
        for g in self.groups:
            s += np.linalg.norm(x[:, g], axis=1).sum()
        return self._penalty * s

    def grad(self, x: np.ndarray) -> np.ndarray:
        res = np.zeros_like(x)
        for g in self.groups:
            norm = np.linalg.norm(x[:, g], axis=1)
            factor = np.true_divide(self._penalty, norm, where=norm > 0)
            res[:, g] = x[:, g] * factor.reshape(-1, 1)
        return res

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1)
        factor = np.true_divide(self._penalty * eta, norm, where=norm > 0)  # avoid dividing by zero
        factor = np.maximum(0, 1 - factor)
        return x * factor.reshape(-1, 1)

    def is_subgradient(self, v: np.ndarray, x: np.ndarray, tol: float) -> bool:
        norm = np.linalg.norm(x, axis=1).reshape(-1, 1)
        not_zero = (norm > 0)
        n = np.true_divide(x, norm, where=not_zero)  # avoid dividing by zero
        u = np.linalg.norm(v - self._penalty * n, axis=1)
        u[~not_zero.ravel()] -= self._penalty
        return np.all(u <= tol)


class SparseGroupLassoPenalty(PenaltyTerm):
    def __init__(self, l1_penalty: float, l2_sqrt_penalty: float, groups = None):
        self.l2_sqrt_penalty = L2SqrtPenalty(l2_sqrt_penalty)
        self.l1_penalty = L1Penalty(l1_penalty)
        self._has_group = False
        self.groups = groups

    @property
    def penalty(self):
        return (self.l1_penalty._penalty, self.l2_sqrt_penalty._penalty)

    @penalty.setter
    def penalty(self, value):
        self.l2_sqrt_penalty.penalty = value
        self.l1_penalty.penalty = value

    @property
    def groups(self):
        return self.l1_penalty.groups if self._has_group else None

    @groups.setter
    def groups(self, value) -> None:
        penalty = self.l1_penalty._penalty

        if value is None:
            self._has_group = False
            self.l1_penalty = L1Penalty(penalty)

        elif self._has_group:
            self.l1_penalty.groups = value

        else:
            self._has_group = True
            self.l1_penalty = L2SqrtPenalty(penalty, value)

    def loss(self, x: np.ndarray) -> float:
        return self.l1_penalty.loss(x) + self.l2_sqrt_penalty.loss(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.l1_penalty.grad(x) + self.l2_sqrt_penalty.grad(x)

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        return self.l2_sqrt_penalty.proximal(self.l1_penalty.proximal(x, eta), eta)

    def is_subgradient(self, v: np.ndarray, x: np.ndarray, tol: float) -> bool:
        lamb1, lamb2 = self.l1_penalty._penalty, self.l2_sqrt_penalty._penalty
        norm = np.linalg.norm(x, axis=1)
        not_zero = (norm > 0)
        n = v - lamb2 * np.true_divide(x, norm.reshape(-1, 1), where=not_zero.reshape(-1, 1))  # avoid dividing by zero
        if not self.l1_penalty.is_subgradient(n[not_zero], x[not_zero], tol):
            return False

        v = v[~not_zero]
        return np.all(np.linalg.norm(v - np.clip(v, -lamb1, lamb1), axis=1) <= lamb2 + tol)


class L2Penalty(PenaltyTerm):
    def loss(self, x: np.ndarray) -> float:
        return self._penalty * x.ravel().dot(x.ravel())

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2 * self._penalty * x

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        return x / (2 * eta * self._penalty + 1)

    def is_subgradient(self, v: np.ndarray, x: np.ndarray, tol: float) -> bool:
        return np.linalg.norm(self.grad(x) - v) <= tol


class InteractionPenalty(PenaltyTerm):
    def loss(self, x: np.ndarray) -> float:
        if x.shape[0] == 1:
            return 0

        xsq = np.square(x)
        M = xsq @ xsq.T
        np.fill_diagonal(M, 0)
        return 0.5 * self._penalty * M.sum()

    def grad(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 1:
            return 0

        xsq = np.square(x)
        col_sq = np.sum(xsq, axis=0)
        grad = (2 * self._penalty) * x * (col_sq - xsq)
        return grad


class HuberPenalty(PenaltyTerm):
    def __init__(self, penalty: float, delta: float = 1e-3):
        assert_positive(delta, 'delta')
        super().__init__(penalty / delta)
        self._delta = delta

    def loss(self, x: np.ndarray) -> float:
        return utils.compute_huber_penalty(x, self._penalty, self._delta)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._penalty * np.clip(x, -self._delta, self._delta)
