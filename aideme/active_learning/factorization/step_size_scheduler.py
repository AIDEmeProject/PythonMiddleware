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
from typing import Callable

from scipy.optimize import OptimizeResult, minimize_scalar

from aideme.utils import assert_positive, assert_positive_integer


__all__ = ['StepSizeScheduler', 'FixedScheduler', 'LineSearchScheduler', 'PowerDecayScheduler']


class StepSizeScheduler:
    def __call__(self, result: OptimizeResult, func: Callable) -> float:
        return self.compute_step_size(result, func)

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        raise NotImplementedError


class FixedScheduler(StepSizeScheduler):
    def __init__(self, step_size: float):
        assert_positive(step_size, 'step_size')
        self.step_size = step_size

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return self.step_size


class LineSearchScheduler(StepSizeScheduler):
    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return minimize_scalar(lambda step: func(result.x - step * result.search_dir), method='Brent').x


class PowerDecayScheduler(StepSizeScheduler):
    def __init__(self, step_size: float, power: float, adapt_every: int = 1):
        assert_positive(step_size, 'step_size')
        assert_positive(power, 'power')
        assert_positive_integer(adapt_every, 'adapt_every')
        self.step_size = step_size
        self.power = power
        self.adapt_every = adapt_every

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return self.step_size / (1 + result.it // self.adapt_every) ** self.power
