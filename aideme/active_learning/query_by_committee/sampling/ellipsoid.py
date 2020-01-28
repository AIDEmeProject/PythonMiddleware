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

from typing import TYPE_CHECKING, Generator, Optional

import numpy as np

if TYPE_CHECKING:
    from .hit_and_run import LinearVersionSpace


class RoundingAlgorithm:
    def __init__(self, max_iter: Optional[int] = None):
        self.max_iter = max_iter if max_iter is not None else float('inf')

    def fit(self, body: LinearVersionSpace) -> Ellipsoid:
        elp = Ellipsoid(body.dim)

        count = 0
        while self._attempt_to_reduce_ellipsoid(elp, body):
            count += 1
            if count >= self.max_iter:
                return elp

        return elp

    def _attempt_to_reduce_ellipsoid(self, elp: Ellipsoid, body: LinearVersionSpace) -> bool:
        return any(self._can_cut(vector, elp, body) for vector in elp.extremes())  # TODO: can we avoid checking the extremes?

    def _can_cut(self, vector: np.ndarray, elp: Ellipsoid, body: LinearVersionSpace) -> bool:
        hyperplane = body.get_separating_oracle(vector)
        return hyperplane is not None and elp.cut(*hyperplane)


class Ellipsoid:
    def __init__(self, dim: int):
        self.dim = dim

        self.center = np.zeros(self.dim)
        self.scale = np.eye(self.dim)

        self.L = np.eye(self.dim)
        self.D = np.ones(self.dim)

    def extremes(self) -> Generator[np.ndarray, None, None]:
        yield self.center

        eig, P = np.linalg.eigh(self.scale + 1e-12 * np.eye(self.dim))  # add small perturbation to diagonal to counter numerical errors

        for i in range(len(eig)):
            if eig[i] <= 0:
                raise RuntimeError("Found non-positive eigenvalue: {}".format(eig[i]))

            factor = np.sqrt(eig[i]) / (self.dim + 1)
            direction = factor * P[:, i]

            yield self.center + direction
            yield self.center - direction

    def cut(self, bias: float, g: np.ndarray) -> bool:
        a_hat = self.L.T.dot(g)
        gamma = np.sqrt(np.square(a_hat).dot(self.D))
        alpha = (g.dot(self.center) - bias) / gamma

        if alpha >= 1:
            raise RuntimeError("Invalid hyperplane: ellipsoid is contained in its positive semi-space (expected the negative one)")

        # shallow cut
        if alpha <= -1.0 / self.dim:
            return False

        p = self.D * a_hat / gamma
        Pg = self.L.dot(p)

        # update center
        tau = (1 + self.dim * alpha) / (self.dim + 1)
        self.center -= tau * Pg

        # update LDL^T
        sigma = 2 * tau / (alpha + 1)
        delta = (1. - alpha * alpha) * ((self.dim * self.dim) / (self.dim * self.dim - 1.))

        beta = self._update_diagonal(p, sigma, delta)
        self._update_cholesky_factor(p, beta)

        # update P
        self.scale -= sigma * (Pg.reshape(-1, 1) @ Pg.reshape(1, -1))
        self.scale *= delta

        return True

    def _update_diagonal(self, p: np.ndarray, sigma: float, delta: float) -> np.ndarray:
        t = np.empty(self.dim + 1)
        t[-1] = 1 - sigma * p.dot(p / self.D)
        t[:-1] = sigma * np.square(p) / self.D
        t = np.cumsum(t[::-1])[::-1]

        self.D *= t[1:]
        beta = -sigma * p / self.D
        self.D /= t[:-1]
        self.D *= delta

        return beta


    def _update_cholesky_factor(self, p: np.ndarray, beta: np.ndarray) -> None:
        v = self.L * p.reshape(1, -1)
        v = np.cumsum(v[:, ::-1], axis=1)[:, ::-1]

        self.L[:, :-1] += v[:, 1:] * beta[:-1].reshape(1, -1)
