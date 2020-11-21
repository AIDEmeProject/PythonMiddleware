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
from typing import Callable, Optional, Union
import warnings

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from aideme.utils import assert_positive, assert_in_range, assert_positive_integer


class OptimizationAlgorithm:
    def __init__(self, step_size: Optional[float] = 1e-3, gtol: float = 1e-4, max_iter: Optional[int] = None, callback: Optional[Callable] = None):
        assert_positive(step_size, 'step_size', allow_none=True)
        assert_positive(gtol, 'gtol')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        self._step_size = step_size
        self._gtol = gtol
        self._max_iter = max_iter if max_iter is not None else np.inf
        self._callback = callback

    def _reset(self) -> None:
        return

    def minimize(self, x0: np.ndarray, func: Callable, grad: Union[bool, Callable]) -> OptimizeResult:
        self._reset()

        result = self.__compute_initial_result_object(x0, grad)
        self.__run_callback(result)

        while not self._converged(result):
            self._advance(result, func, grad)
            self.__run_callback(result)

        result.fun = func(result.x)

        if not result.success:
            warnings.warn("Optimization routine did not converge: max iter reached.\n{}".format(result))

        return result

    def __compute_initial_result_object(self, x0: np.ndarray, grad: Callable) -> OptimizeResult:
        result = OptimizeResult()
        result.x = x0.copy()
        result.grad = grad(result.x)
        result.it = 1
        result.step = None
        result.success = False
        return result

    def __run_callback(self, result: OptimizeResult) -> None:
        if self._callback is not None:
            self._callback(result.x)

    def _converged(self, result: OptimizeResult) -> bool:
        if np.linalg.norm(result.grad) < self._gtol:
            result.success = True
            return True

        return result.it > self._max_iter

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> None:
        search_dir = self._compute_search_dir(result)
        result.step = self._compute_step_size(result.x, search_dir, func)
        result.x -= result.step * search_dir
        result.grad = grad(result.x)
        result.it += 1

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        """
        :param result: the current state of optimization (x, grad, etc)
        :return: the next optimizer in this iterative process
        """
        raise NotImplementedError

    def _compute_step_size(self, x: np.ndarray, search_dir: np.ndarray, func: Callable):
        return self._step_size


class Adam(OptimizationAlgorithm):
    def __init__(self, step_size: float = 1e-3, gtol: float = 1e-4, max_iter: Optional[int] = None, callback: Optional[Callable] = None,
                 adapt_step_size: bool = False, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(step_size=step_size, gtol=gtol, max_iter=max_iter, callback=callback)
        assert_in_range(beta1, 'beta1', 0, 1)
        assert_in_range(beta2, 'beta2', 0, 1)
        assert_positive(epsilon, 'epsilon')

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._adapt_step_size = adapt_step_size

        self._beta1_t = 1.0
        self._beta2_t = 1.0
        self._mt = self._vt = 0
        self._t = 0

    def _reset(self):
        self._beta1_t = 1.0
        self._beta2_t = 1.0
        self._mt = self._vt = 0
        self._t = 0

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        self._t += 1

        self._mt = self._beta1 * self._mt + (1 - self._beta1) * result.grad
        self._vt = self._beta2 * self._vt + (1 - self._beta2) * np.square(result.grad)

        self._beta1_t *= self._beta1
        self._beta2_t *= self._beta2

        m_hat = self._mt / (1 - self._beta1_t)
        v_hat = self._vt / (1 - self._beta2_t)

        return m_hat / (np.sqrt(v_hat) + self._epsilon)

    def _compute_step_size(self, x: np.ndarray, search_dir: np.ndarray, func: Callable):
        return self._step_size / np.sqrt(self._t) if self._adapt_step_size else self._step_size


class GradientDescent(OptimizationAlgorithm):
    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        return result.grad

    def _compute_step_size(self, x: np.ndarray, search_dir: np.ndarray, func: Callable):
        if self._step_size is None:
            return minimize_scalar(lambda step: func(x - step * search_dir), method='Brent').x

        return self._step_size


class NoisyGradientDescent(GradientDescent):
    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        search_dir = result.grad
        noise = np.random.normal(size=search_dir.shape)
        noise /= np.linalg.norm(noise)
        return search_dir + noise


class BFGS(OptimizationAlgorithm):
    def minimize(self, x0: np.ndarray, func: Callable, grad: Union[bool, Callable]) -> OptimizeResult:
        jac = lambda x: grad(x).ravel()
        return minimize(func, x0, jac=jac, callback=self._callback, options={'gtol': self._gtol, 'maxiter': self._max_iter})


class ProximalGradientDescent(OptimizationAlgorithm):
    # TODO: implement this
    pass
