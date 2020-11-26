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

import warnings
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult, minimize, minimize_scalar

from aideme.utils import assert_positive, assert_in_range, assert_positive_integer

if TYPE_CHECKING:
    from .penalty import PenaltyTerm


class OptimizationAlgorithm:
    def __init__(self, gtol: float = 1e-4, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        assert_positive(gtol, 'gtol')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        self._gtol = gtol
        self._max_iter = max_iter if max_iter is not None else np.inf
        self._callback = callback
        self._verbose = verbose

    def _reset(self) -> None:
        return

    def minimize(self, x0: np.ndarray, func: Callable, grad: Union[bool, Callable]) -> OptimizeResult:
        self._reset()

        result = self._build_initial_result_object(x0)
        self.__update_result_object(result, grad)
        self.__run_callback(result)

        while not self._converged(result):
            self._advance(result, func, grad)
            self.__update_result_object(result, grad)
            self.__run_callback(result)

        result.fun = self._compute_opt_result(func, result.x)

        if self._verbose and not result.success:
            warnings.warn("Optimization routine did not converge: max iter reached.\n{}".format(result))

        return result

    @staticmethod
    def _build_initial_result_object(x0: np.ndarray) -> OptimizeResult:
        result = OptimizeResult()
        result.x = x0.copy()
        result.it = 0
        result.success = False
        return result

    def __update_result_object(self, result: OptimizeResult, grad: Callable):
        result.it += 1
        result.grad = grad(result.x)

    def __run_callback(self, result: OptimizeResult) -> None:
        if self._callback is not None:
            self._callback(result.x)

    def _converged(self, result: OptimizeResult) -> bool:
        if np.linalg.norm(result.grad) < self._gtol:
            result.success = True
            return True

        return result.it > self._max_iter

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> None:
        raise NotImplementedError

    def _compute_opt_result(self, func: Callable, x_opt: np.ndarray) -> float:
        return func(x_opt)


class BFGS(OptimizationAlgorithm):
    def minimize(self, x0: np.ndarray, func: Callable, grad: Union[bool, Callable]) -> OptimizeResult:
        jac = lambda x: grad(x).ravel()
        return minimize(func, x0, jac=jac, callback=self._callback, options={'gtol': self._gtol, 'maxiter': self._max_iter})


class SearchDirectionOptimizer(OptimizationAlgorithm):
    def __init__(self, step_size: Optional[float] = 1e-3, gtol: float = 1e-4, max_iter: Optional[int] = None,
                 callback: Optional[Callable] = None, verbose: bool = False):
        assert_positive(step_size, 'step_size', allow_none=True)

        super().__init__(gtol=gtol, max_iter=max_iter, callback=callback, verbose=verbose)
        self._step_size = step_size

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> None:
        result.search_dir = self._compute_search_dir(result)
        result.step = self._compute_step_size(result, func)
        result.x -= result.step * result.search_dir

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        """
        :param result: the current state of optimization (x, grad, etc)
        :return: the next optimizer in this iterative process
        """
        raise NotImplementedError

    def _compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return self._step_size


class GradientDescent(SearchDirectionOptimizer):
    def __init__(self, batch_size: Optional[int] = None, step_size: float = 1e-3, adapt_step_size: float = False,
                 gtol: float = 1e-4, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        super().__init__(step_size=step_size, gtol=gtol, max_iter=max_iter, callback=callback, verbose=verbose)
        assert_positive_integer(batch_size, 'batch_size', allow_none=True)
        self._adapt_step_size = adapt_step_size
        self.batch_size = batch_size

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        return result.grad

    def _compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        if self._step_size is None:
            return minimize_scalar(lambda step: func(result.x - step * result.search_dir), method='Brent').x

        if self._adapt_step_size:
            return self._step_size / result.it

        return self._step_size


class NoisyGradientDescent(GradientDescent):
    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        search_dir = result.grad
        noise = np.random.normal(size=search_dir.shape)
        noise /= np.linalg.norm(noise)
        return search_dir + noise


class ProximalGradientDescent(OptimizationAlgorithm):
    def __init__(self, penalty_term: Optional[PenaltyTerm] = None, step_size: float = 1e-3, remove_bias_column: bool = False,
                 gtol: float = 1e-4, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        assert_positive(step_size, 'step_size')

        super().__init__(gtol=gtol, max_iter=max_iter, callback=callback, verbose=verbose)
        self._step_size = step_size
        self.penalty_term = penalty_term
        self.remove_bias_column = remove_bias_column

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> None:
        next_x = result.x - self._step_size * result.grad
        if self.remove_bias_column:
            next_x[:, :-1] = self.penalty_term.proximal(next_x[:, :-1], self._step_size)
        else:
            next_x = self.penalty_term.proximal(next_x, self._step_size)
        result.x = next_x

    def _compute_opt_result(self, func: Callable, x_opt: np.ndarray) -> float:
        return func(x_opt) + self.penalty_term.loss(x_opt)


class Adam(SearchDirectionOptimizer):
    def __init__(self, batch_size: Optional[int] = None, step_size: float = 1e-3, adapt_step_size: bool = False,
                 gtol: float = 1e-4, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(step_size=step_size, gtol=gtol, max_iter=max_iter, callback=callback, verbose=verbose)
        assert_in_range(beta1, 'beta1', 0, 1)
        assert_in_range(beta2, 'beta2', 0, 1)
        assert_positive(epsilon, 'epsilon')

        assert_positive_integer(batch_size, 'batch_size', allow_none=True)
        self.batch_size = batch_size

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._adapt_step_size = adapt_step_size

        self._beta1_t = 1.0
        self._beta2_t = 1.0
        self._mt = self._vt = 0

    def _reset(self):
        self._beta1_t = 1.0
        self._beta2_t = 1.0
        self._mt = self._vt = 0

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        self._mt = self._beta1 * self._mt + (1 - self._beta1) * result.grad
        self._vt = self._beta2 * self._vt + (1 - self._beta2) * np.square(result.grad)

        self._beta1_t *= self._beta1
        self._beta2_t *= self._beta2

        m_hat = self._mt / (1 - self._beta1_t)
        v_hat = self._vt / (1 - self._beta2_t)

        return m_hat / (np.sqrt(v_hat) + self._epsilon)

    def _compute_step_size(self, result: OptimizeResult, func: Callable):
        return self._step_size / np.sqrt(result.it) if self._adapt_step_size else self._step_size
