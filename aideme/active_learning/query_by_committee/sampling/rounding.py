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

from .ellipsoid import Ellipsoid
from aideme.utils import assert_positive_integer

if TYPE_CHECKING:
    from .hit_and_run import LinearVersionSpace
    STRATEGY_TYPE = Callable[[Ellipsoid, LinearVersionSpace], bool]


class RoundingAlgorithm:
    def __init__(self, max_iter: Optional[int] = None, strategy: str = 'default'):
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        self.max_iter = max_iter if max_iter is not None else float('inf')
        self.strategy, self.compute_scale_matrix = self.__get_strategy(strategy)

    @staticmethod
    def __get_strategy(strategy: str) -> Tuple[STRATEGY_TYPE, bool]:
        strategy = strategy.upper()
        if strategy == 'DEFAULT':
            return diagonalization_strategy, True
        if strategy == 'OPT':
            return new_strategy, False
        raise ValueError("Unknown strategy {}. Possible values are: 'default', 'opt'.")

    def fit(self, body: LinearVersionSpace) -> Ellipsoid:
        elp = Ellipsoid(body.dim, compute_scale_matrix=self.compute_scale_matrix)

        count = 0
        while self.strategy(elp, body):
            count += 1
            if count >= self.max_iter:
                # TODO: log this
                return elp
        # TODO: log avg_time and count

        return elp


def diagonalization_strategy(elp: Ellipsoid, body: LinearVersionSpace) -> bool:
    for vector in elp.extremes():
        hyperplane = body.get_separating_oracle(vector)

        if hyperplane is not None and elp.cut(*hyperplane):
            return True

    return False


def new_strategy(elp: Ellipsoid, body: LinearVersionSpace) -> bool:
    # find best cut
    alphas = elp.compute_alpha(body.A)
    idx_max = np.argmax(alphas)
    alpha_max = alphas[idx_max]

    # stop criteria: gamma (= -alpha) >= threshold (= 1 / (n+1)*sqrt(n))
    threshold = 1 / ((elp.dim + 1) * np.sqrt(elp.dim))
    # TODO: can it happen that |z| >= 1 in the end?

    return alpha_max >= -threshold and elp.cut(0, body.A[idx_max])
