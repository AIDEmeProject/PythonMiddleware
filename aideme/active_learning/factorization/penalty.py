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

import aideme.active_learning.factorization.utils as utils
from aideme.utils import assert_positive


class PenaltyTerm:
    def __init__(self, penalty: float):
        assert_positive(penalty, 'penalty')
        self._penalty = penalty

    def loss(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        return None


class L1Penalty(PenaltyTerm):
    def loss(self, x: np.ndarray) -> float:
        return self._penalty * np.abs(x).sum()

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self._penalty * np.sign(x)

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - self._penalty * eta, 0)


class L2SqrtPenalty(PenaltyTerm):
    def loss(self, x: np.ndarray) -> float:
        return self._penalty * np.linalg.norm(x, axis=1).sum()

    def grad(self, x: np.ndarray) -> np.ndarray:
        norm = self._penalty / np.linalg.norm(x, axis=1)
        return x * norm.reshape(-1, 1)

    def proximal(self, x: np.ndarray, eta: float) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1)
        factor = np.maximum(0, 1 - self._penalty * eta / norm)
        return x * factor.reshape(-1, 1)


class L2Penalty(PenaltyTerm):
    def loss(self, x: np.ndarray) -> float:
        return self._penalty * x.ravel().dot(x.ravel())

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2 * self._penalty * x


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
