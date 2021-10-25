#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.
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
        self._add_grad_to_result = True

    def _reset(self) -> None:
        return

    def minimize(self, x0: np.ndarray, func: Callable, grad: Callable) -> OptimizeResult:
        self._reset()

        result = self.__build_initial_result_object()
        new_x = x0.copy()
        self.__process_new_iter(grad, new_x, result)

        while not result.success and result.it < self._max_iter:
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

    def __update_result_object(self, result: OptimizeResult, new_x: np.ndarray, grad: Callable):
        result.it += 1
        result.prev, result.x = result.x, new_x
        if self._add_grad_to_result:
            result.grad = grad(new_x) if result.it < self._max_iter else None

    def __converged(self, result: OptimizeResult) -> bool:
        if result.prev is not None and np.linalg.norm(result.x - result.prev) < self._rel_tol * np.linalg.norm(result.x):
            return True

        return self._gradient_converged(result, self._gtol) if result.get('grad', None) is not None else False

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
                 step_size: Optional[float] = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 1, exp_decay: float = 0,
                 gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        assert_positive_integer(batch_size, 'batch_size', allow_none=True)
        super().__init__(gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)
        self.batch_size = batch_size
        self._step_size_scheduler = self.__get_step_size_scheduler(step_size, adapt_step_size, adapt_every, power, exp_decay)

    @staticmethod
    def __get_step_size_scheduler(step_size: Optional[float], adapt_step_size: bool, adapt_every: int, power: float, exp_decay: float) -> StepSizeScheduler:
        if step_size is None:
            return LineSearchScheduler()

        if adapt_step_size:
            return ExponentialDecayScheduler(step_size=step_size, decay=exp_decay, adapt_every=adapt_every) if exp_decay > 0 else PowerDecayScheduler(step_size=step_size, power=power, adapt_every=adapt_every)

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
    def __init__(self, batch_size: Optional[int] = None, momentum: float = 0.9,
                 step_size: float = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 1, exp_decay: float = 0,
                 gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        assert_in_range(momentum, 'momentum', 0, 1)
        super().__init__(batch_size=batch_size, step_size=step_size,
                         adapt_step_size=adapt_step_size, adapt_every=adapt_every, power=power, exp_decay=exp_decay,
                         gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)
        self.momentum = momentum
        self.v = 0

    def _reset(self) -> None:
        self.v = 0

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        self.v = self.momentum * self.v + (1 - self.momentum) * result.grad
        return self.v


class NoisyGradientDescent(SearchDirectionOptimizer):
    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        search_dir = result.grad
        noise = np.random.normal(size=search_dir.shape)
        noise /= np.linalg.norm(noise)
        return search_dir + noise


class ProximalGradientDescent(SearchDirectionOptimizer):
    def __init__(self, penalty_term: Optional[PenaltyTerm] = None, batch_size: Optional[int] = None,
                 step_size: float = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 1, exp_decay: float = 0,
                 gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        super().__init__(batch_size=batch_size, step_size=step_size,
                         adapt_step_size=adapt_step_size, adapt_every=adapt_every, power=power, exp_decay=exp_decay,
                         gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)

        self.penalty_term = penalty_term
        self.remove_bias_column = False

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> np.ndarray:
        step = self._step_size_scheduler(result, func)
        return self._proximal_step(result.x, result.grad, step)

    def _proximal_step(self, x: np.ndarray, grad: np.ndarray, step: float) -> np.ndarray:
        next_x = x - step * grad
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


class FISTA(ProximalGradientDescent):
    def __init__(self, penalty_term: Optional[PenaltyTerm] = None, batch_size: Optional[int] = None,
                 step_size: float = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 1, exp_decay: float = 0,
                 gtol: float = 1e-4, rel_tol: float = 0, max_iter: Optional[int] = None, callback: Optional[Callable] = None, verbose: bool = False):
        super().__init__(penalty_term=penalty_term, batch_size=batch_size, step_size=step_size, adapt_step_size=adapt_step_size,
                         adapt_every=adapt_every, power=power, exp_decay=exp_decay, gtol=gtol, rel_tol=rel_tol, max_iter=max_iter,
                         callback=callback, verbose=verbose)
        self.y = None
        self.theta = 1.
        self._add_grad_to_result = False

    def _reset(self) -> None:
        self.y = None
        self.theta = 1.

    def _advance(self, result: OptimizeResult, func: Callable, grad: Callable) -> np.ndarray:
        if self.y is None:
            self.y = result.x

        # update x
        step = self._step_size_scheduler(result, func)
        next_x = self._proximal_step(self.y, grad(self.y), step)

        # Update momentum terms y and theta
        next_th = (1 + np.sqrt(1 + 4 * self.theta * self.theta)) / 2
        self.y = next_x + ((self.theta - 1) / next_th) * (next_x - result.x)
        self.theta = next_th

        return next_x


class Adam(SearchDirectionOptimizer):
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, batch_size: Optional[int] = None,
                 step_size: float = 1e-3, adapt_step_size: bool = False, adapt_every: int = 1, power: float = 0.5, exp_decay: float = 0,
                 gtol: float = 1e-4, max_iter: Optional[int] = None, rel_tol: float = 0, callback: Optional[Callable] = None, verbose: bool = False):
        assert_in_range(beta1, 'beta1', 0, 1)
        assert_in_range(beta2, 'beta2', 0, 1)
        assert_positive(epsilon, 'epsilon')
        super().__init__(batch_size=batch_size, step_size=step_size,
                         adapt_step_size=adapt_step_size, adapt_every=adapt_every, power=power, exp_decay=exp_decay,
                         gtol=gtol, rel_tol=rel_tol, max_iter=max_iter, callback=callback, verbose=verbose)

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._mt = self._vt = 0

    def _reset(self):
        self._mt = self._vt = 0

    def _compute_search_dir(self, result: OptimizeResult) -> np.ndarray:
        self._mt = self._beta1 * self._mt + (1 - self._beta1) * result.grad
        self._vt = self._beta2 * self._vt + (1 - self._beta2) * np.square(result.grad)

        it = result.it + 1
        beta1_t = self._beta1 ** it
        beta2_t = self._beta2 ** it

        m_hat = self._mt / (1 - beta1_t)
        v_hat = self._vt / (1 - beta2_t)

        return m_hat / (np.sqrt(v_hat) + self._epsilon)
