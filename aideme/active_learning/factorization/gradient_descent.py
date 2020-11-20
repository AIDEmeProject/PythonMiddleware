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
from typing import Callable, Optional, Dict, List

import warnings

import numpy as np
import scipy.optimize

from aideme import assert_positive, assert_positive_integer


class ResultObject:
    def __init__(self, x: np.ndarray, fun: float, grad: np.ndarray, step: float = None, iters: int = 0):
        self.x = x
        self.fun = fun
        self.grad = grad
        self.step = step
        self.grad_norm = np.linalg.norm(grad)
        self.converged = False
        self.iters = iters

    def __repr__(self):
        return "Result\nx={}\nfun={}\ngrad={}\ngrad_norm={}\nstep={}\nconverged={}\niters={}".format(
            self.x, self.fun, self.grad, self.grad_norm, self.step, self.converged, self.iters
        )


class GradientDescentOptimizer:
    def __init__(self, add_noise: bool = False, grad_norm_threshold: float = 1e-4,
                 max_iter: Optional[int] = None, step_size: Optional[float] = None):
        assert_positive(grad_norm_threshold, 'grad_norm_threshold')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)
        assert_positive(step_size, 'step_size', allow_none=True)

        self.add_noise = add_noise
        self.step_optimizer = self.__get_step_optimizer(step_size)
        self.grad_norm_threshold = grad_norm_threshold
        self.max_iter = max_iter if max_iter else np.inf

    def __get_step_optimizer(self, step_size: Optional[float] = None) -> Callable:
        if step_size is None:
            return lambda func, x, dir: scipy.optimize.minimize_scalar(lambda step: func(x - step * dir), method='Brent').x

        return lambda func, x, dir: step_size

    def minimize(self, x0: np.ndarray, func: Callable, func_and_grad: Callable, func_threshold: float = -np.inf) -> ResultObject:
        x = np.array(x0, copy=True)
        fval, grad = func_and_grad(x)
        prev_result, result = None, ResultObject(x, fval, grad)

        while not self._converged(result):
            prev_result, result = result, self._advance(result, func, func_and_grad)
            if result.iters >= self.max_iter or result.fun < func_threshold:
                return result

        result.converged = True
        return result

    def _converged(self, result: ResultObject) -> bool:
        return result.grad_norm <= self.grad_norm_threshold

    def _advance(self, result: ResultObject, func: Callable, func_and_grad: Callable) -> ResultObject:
        search_dir = self.__get_search_dir(result)
        step = self.step_optimizer(func, result.x, search_dir)
        x = result.x - step * search_dir
        fval, grad = func_and_grad(x)
        return ResultObject(x, fval, grad, step, result.iters + 1)

    def __get_search_dir(self, result: ResultObject):
        search_dir = result.grad.copy()
        if self.add_noise:
            noise = np.random.normal(size=search_dir.shape)
            noise /= np.linalg.norm(noise)
            search_dir += noise
        return search_dir


class ProximalGradientDescentOptimizer:
    def __init__(self, conv_threshold: float = 1e-3, max_iter: Optional[int] = None,
                 step_size_method: str = 'backtrack', step_size_params: Optional[Dict] = None,
                 callbacks: Optional[List[Callable]] = None):
        assert_positive(conv_threshold, 'conv_threshold')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        if step_size_params is None:
            step_size_params = {}

        if callbacks is None:
            callbacks = []

        self.conv_threshold = conv_threshold
        self.step_optimizer = self.__get_step_optimizer(step_size_method, step_size_params)
        self.max_iter = max_iter if max_iter else np.inf
        self.callbacks = callbacks

    @staticmethod
    def __get_step_optimizer(step_size_method: str, step_size_params: Dict) -> Callable:
        step_size_method = step_size_method.upper()

        if step_size_method == 'FIXED':
            return fixed_size_step(**step_size_params)

        if step_size_method == 'BACKTRACK':
            return line_search_backtracking(**step_size_params)

        raise ValueError("Unknown step optimized method '{}'".format(step_size_method))

    def minimize(self, x0: np.ndarray, f: Callable, fprime: Callable, g: Callable, proxg: Callable) -> ResultObject:
        x = np.array(x0, copy=True)
        fval, fgrad = fprime(x)
        prev_result, result = None, ResultObject(x, fval + g(x), fgrad)

        while not self._converged(prev_result, result):
            prev_result, result = result, self._advance(result, f, fprime, g, proxg)
            self._run_callbacks(prev_result, result)

        return result

    def _converged(self, prev_result: ResultObject, result: ResultObject) -> bool:
        if prev_result is None:  # First iteration
            return False

        if np.linalg.norm(result.x - prev_result.x) < self.conv_threshold:
            result.converged = True
            return True

        return result.iters > self.max_iter

    def _advance(self, result: ResultObject, f: Callable, fprime: Callable, g: Callable, proxg: Callable) -> ResultObject:
        step = self.step_optimizer(result, f, proxg)
        x = proxg(result.x - step * result.grad, step)
        fval, fgrad = fprime(x)
        return ResultObject(x, fval + g(x), fgrad, step, result.iters + 1)

    def _run_callbacks(self, prev_result: ResultObject, result: ResultObject) -> None:
        for callback in self.callbacks:
            callback(prev_result, result)


def l1_penalty_func_and_prox(penalty: float, has_bias: bool = False):
    assert_positive(penalty, 'penalty')

    def g(x):
        if has_bias:
            x = x[:, :-1]
        return penalty * np.abs(x).sum()

    def prox(x, t):
        p = np.sign(x) * np.maximum(np.abs(x) - penalty * t, 0)
        if has_bias:
            p[:, -1] = x[:, -1]
        return p

    return g, prox


def fixed_size_step(step_size):
    assert_positive(step_size, 'step_size')
    return lambda result, func, prox: step_size


def line_search_backtracking(beta: float = 0.99, max_iter: float = 1000):
    assert_positive(beta, 'beta')
    assert_positive_integer(max_iter, 'max_iter_line_search')

    def f(result, func, prox):
        x, fx, gx = result.x, result.fun, result.grad

        alpha = 1.0 if result.step is None else result.step
        it = 0
        x_new = prox(x - alpha * gx, alpha)
        diff = x_new - x
        while func(x_new) > fx + prod(gx, diff) + 0.5 * prod(diff, diff) / alpha and it < max_iter:
            alpha *= beta
            it += 1
            x_new = prox(x - alpha * gx, alpha)
            diff = x_new - x

        if it == max_iter:
            warnings.warn("Line-search did not converge: max iter reached.")

        return alpha

    return f

def prod(x, y):
    return x.ravel().dot(y.ravel())
