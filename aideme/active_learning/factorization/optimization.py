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
from scipy.optimize import OptimizeResult, minimize

from aideme.utils import assert_positive, assert_in_range, assert_positive_integer, assert_non_negative
from .step_size_scheduler import *

if TYPE_CHECKING:
    from .penalty import PenaltyTerm


class OptimizationAlgorithm:
    def __init__(self, gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        assert_non_negative(gtol, 'gtol')
        assert_non_negative(rel_tol, 'rel_tol')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        self._gtol = gtol
        self._rel_tol = rel_tol
        self._max_iter = max_iter if max_iter is not None else np.inf
        self._callback = callback
        self._verbose = verbose

    def _reset(self) -> None:
        return

    def minimize(self, x0: np.ndarray, func: Callable, grad: Union[bool, Callable]) -> OptimizeResult:
        self._reset()

        result = self.__build_initial_result_object()
        new_x = x0.copy()
        self.__process_new_iter(grad, new_x, result)

        while not result.success and result.it <= self._max_iter:
            new_x = self._advance(result, func, grad)
            self.__process_new_iter(grad, new_x, result)

        result.fun = func(result.x)

        if self._verbose and not result.success:
            warnings.warn("Optimization routine did not converge: max iter reached.\n{}".format(result))

        return result

    @staticmethod
    def __build_initial_result_object() -> OptimizeResult:
        result = OptimizeResult()
        result.x = None
        result.it = -1
        result.success = False
        return result

    def __process_new_iter(self, grad, new_x, result):
        self.__update_result_object(result, new_x, grad)
        result.success = self.__converged(result)
        self.__run_callback(result)

    @staticmethod
    def __update_result_object(result: OptimizeResult, new_x: np.ndarray, grad: Callable):
        result.it += 1
        result.prev, result.x = result.x, new_x
        result.grad = grad(new_x)

    def __converged(self, result: OptimizeResult) -> bool:
        if result.prev is not None and np.linalg.norm(result.x - result.prev) < self._rel_tol * np.linalg.norm(result.x):
            return True

        return self._gradient_converged(result, self._gtol)

    def __run_callback(self, result: OptimizeResult) -> None:
        if self._callback is not None:
            self._callback(result.x)

    def _gradient_converged(self, result: OptimizeResult, tol: float) -> bool:
        return np.linalg.norm(result.grad) < tol

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> np.ndarray:
        raise NotImplementedError


class BFGS(OptimizationAlgorithm):
    def minimize(self, x0: np.ndarray, func: Callable, grad: Union[bool, Callable]) -> OptimizeResult:
        jac = lambda x: grad(x).ravel()
        return minimize(func, x0, jac=jac, callback=self._callback, options={'gtol': self._gtol, 'maxiter': self._max_iter})


class SearchDirectionOptimizer(OptimizationAlgorithm):
    def __init__(self, batch_size: Optional[int] = None,
                 step_size: Optional[float] = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 1,
                 gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        assert_positive_integer(batch_size, 'batch_size', allow_none=True)
        super().__init__(gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)
        self.batch_size = batch_size
        self._step_size_scheduler = self.__get_step_size_scheduler(step_size, adapt_step_size, adapt_every, power)

    @staticmethod
    def __get_step_size_scheduler(step_size: Optional[float], adapt_step_size: bool, adapt_every: int, power: float) -> StepSizeScheduler:
        if step_size is None:
            return LineSearchScheduler()

        if adapt_step_size:
            return PowerDecayScheduler(step_size=step_size, power=power, adapt_every=adapt_every)

        return FixedScheduler(step_size)

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> np.ndarray:
        result.search_dir = self._compute_search_dir(result)
        step = self._step_size_scheduler(result, func)
        return result.x - step * result.search_dir

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        """
        :param result: the current state of optimization (x, grad, etc)
        :return: the next optimizer in this iterative process
        """
        raise NotImplementedError


class GradientDescent(SearchDirectionOptimizer):
    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        return result.grad


class NoisyGradientDescent(SearchDirectionOptimizer):
    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        search_dir = result.grad
        noise = np.random.normal(size=search_dir.shape)
        noise /= np.linalg.norm(noise)
        return search_dir + noise


class ProximalGradientDescent(SearchDirectionOptimizer):
    def __init__(self, penalty_term: Optional[PenaltyTerm] = None, batch_size: Optional[int] = None,
                 step_size: float = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 1,
                 gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        super().__init__(batch_size=batch_size, step_size=step_size, adapt_step_size=adapt_step_size, adapt_every=adapt_every, power=power,
                         gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)

        self.penalty_term = penalty_term
        self.remove_bias_column = False

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> np.ndarray:
        step = self._step_size_scheduler(result, func)
        next_x = result.x - step * result.grad
        _, next_weights = self.__separate_bias(next_x)
        np.copyto(next_weights, self.penalty_term.proximal(next_weights, step))
        return next_x

    def _gradient_converged(self, result: OptimizeResult, tol: float) -> bool:
        grad_b, grad_w = self.__separate_bias(result.grad)
        _, w = self.__separate_bias(result.x)
        return np.linalg.norm(grad_b) <= tol and self.penalty_term.is_subgradient(-grad_w, w, tol)

    def __separate_bias(self, x: np.ndarray):
        bias = x[:, -1] if self.remove_bias_column else 0
        x_wo_bias = x[:, :-1] if self.remove_bias_column else x
        return bias, x_wo_bias


class Adam(SearchDirectionOptimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, batch_size: Optional[int] = None,
                 step_size: float = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 0.5,
                 gtol: float = 1e-4, max_iter: Optional[int] = None, rel_tol: float = 0, callback: Optional[Callable] = None, verbose: bool = False):
        assert_in_range(beta1, 'beta1', 0, 1)
        assert_in_range(beta2, 'beta2', 0, 1)
        assert_positive(epsilon, 'epsilon')
        super().__init__(batch_size=batch_size, step_size=step_size, adapt_step_size=adapt_step_size, adapt_every=adapt_every, power=power,
                         gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

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
