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

from typing import TYPE_CHECKING, Optional, Callable, Tuple

import numpy as np

from aideme.utils import assert_positive_integer, metric_logger
from .ellipsoid import Ellipsoid

if TYPE_CHECKING:
    from .hit_and_run import LinearVersionSpace
    Strategy = Callable[[Ellipsoid, LinearVersionSpace], bool]
    HyperPlane = Tuple[float, np.ndarray]


class RoundingAlgorithm:
    def __init__(self, max_iter: Optional[int] = None, strategy: str = 'default', z_cut: bool = False):
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        self.max_iter = max_iter if max_iter is not None else float('inf')
        self.strategy, self.compute_scale_matrix = self.__get_strategy(strategy, z_cut)

    @staticmethod
    def __get_strategy(strategy: str, z_cut: bool) -> Tuple[Strategy, bool]:
        strategy = strategy.upper()
        if strategy == 'DEFAULT':
            return diagonalization_strategy, True
        if strategy == 'OPT':
            return OptimizedStrategy(z_cut=z_cut), False
        raise ValueError("Unknown strategy {}. Possible values are: 'default', 'opt'.")

    @metric_logger.log_execution_time('rounding_fit_time')
    def fit(self, body: LinearVersionSpace, elp: Optional[Ellipsoid] = None) -> Ellipsoid:
        if elp is None:
            elp = Ellipsoid(body.dim, compute_scale_matrix=self.compute_scale_matrix)

        count = 0
        while count < self.max_iter and self.strategy(elp, body):
            count += 1

        metric_logger.log_metric('rounding_iter', count)

        return elp


def diagonalization_strategy(elp: Ellipsoid, body: LinearVersionSpace) -> bool:
    for vector in elp.extremes():
        hyperplane = body.get_separating_oracle(vector)

        if hyperplane is not None and elp.cut(*hyperplane):
            return True

    return False


class OptimizedStrategy:
    def __init__(self, z_cut: bool = False):
        self.z_cut = z_cut

    def __call__(self, elp: Ellipsoid, body: LinearVersionSpace) -> bool:
        alpha, hyperplane = self._get_alpha_cut(elp, body)

        if self.z_cut and alpha != 0:
            alpha_z, hyperplane_z = self._get_z_cut(elp)
            if alpha_z > alpha:
                alpha, hyperplane = alpha_z, hyperplane_z

        threshold = 1 / ((elp.dim + 1) * np.sqrt(elp.dim))
        if -alpha >= threshold:
            return False

        elp.cut(*hyperplane)
        return True

    def _get_alpha_cut(self, elp: Ellipsoid, body: LinearVersionSpace) -> Tuple[float, HyperPlane]:
        alphas = elp.compute_alpha(body.A)
        idx_max = np.argmax(alphas)
        return alphas[idx_max], (0, body.A[idx_max])

    def _get_z_cut(self, elp: Ellipsoid) -> Tuple[float, HyperPlane]:
        hyperplane = np.linalg.norm(elp.center), elp.center
        return elp.compute_alpha_single(*hyperplane), hyperplane
