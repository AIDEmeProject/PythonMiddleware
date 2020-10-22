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
from typing import Callable, Optional

import numpy as np
import scipy.optimize

from aideme import assert_positive, assert_positive_integer
import scipy.optimize


class ResultObject:
    def __init__(self, x: np.ndarray, fun: float, grad: np.ndarray, iters: int = 0):
        self.x = x
        self.fun = fun
        self.grad = grad
        self.grad_norm = np.linalg.norm(grad)
        self.converged = False
        self.iters = iters

    def __repr__(self):
        return "Result\nx={}\nfun={}\ngrad={}\ngrad_norm={}\nconverged={}\niters={}".format(self.x, self.fun, self.grad,
                                                                                            self.grad_norm,
                                                                                            self.converged, self.iters)


class GradientDescentOptimizer:
    def __init__(self, grad_norm_threshold: float = 1e-4, rel_tol: float = 1e-4,
                 max_iter: Optional[int] = None, step_size: Optional[float] = None):
        assert_positive(grad_norm_threshold, 'grad_norm_threshold')
        assert_positive(rel_tol, 'rel_tol')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)
        assert_positive(step_size, 'step_size', allow_none=True)

        self.step_optimizer = self.__get_step_optimizer(step_size)
        self.grad_norm_threshold = grad_norm_threshold
        self.rel_tol = rel_tol
        self.max_iter = max_iter if max_iter else np.inf

    def __get_step_optimizer(self, step_size: Optional[float] = None) -> Callable:
        if step_size is not None:
            return lambda func, result: step_size

        return lambda func, result: scipy.optimize.minimize_scalar(lambda step: func(self._retraction(step, result)), method='Brent').x

    def optimize(self, x0: np.ndarray, func: Callable, grad: Callable, func_threshold: float = -np.inf) -> ResultObject:
        x = np.array(x0, copy=True)
        prev_result, result = None, ResultObject(x, func(x), grad(x))

        while not self._converged(result, prev_result):
            prev_result, result = result, self._advance(result, func, grad)

            if result.iters >= self.max_iter or result.fun < func_threshold:
                return result

        result.converged = True
        return result

    def _converged(self, result: ResultObject, prev_result: Optional[ResultObject] = None) -> bool:
        if result.grad_norm <= self.grad_norm_threshold:
            return True

        if prev_result is None:  # first iteration
            return False

        return (prev_result.fun - result.fun) < self.rel_tol * prev_result.fun

    def _advance(self, result: ResultObject, func: Callable, grad: Callable) -> ResultObject:
        step = self.step_optimizer(func, result)
        x = self._retraction(step, result)
        return ResultObject(x, func(x), grad(x), result.iters + 1)

    @staticmethod
    def _retraction(step: float, result: ResultObject) -> np.ndarray:
        return result.x - step * result.grad
