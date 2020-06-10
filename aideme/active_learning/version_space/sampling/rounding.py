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
import scipy.optimize

from aideme.utils import assert_positive_integer, assert_positive, metric_logger
from .ellipsoid import Ellipsoid

if TYPE_CHECKING:
    from .polyhedral_cone import BoundedPolyhedralCone
    from aideme.utils import HyperPlane
    Strategy = Callable[[Ellipsoid, BoundedPolyhedralCone], bool]


class RoundingAlgorithm:
    def __init__(self, max_iter: Optional[int] = None, strategy: str = 'opt', z_cut: bool = True, sphere_cuts: bool = False, sphere_cuts_optimizer = None):
        """
        :param max_iter: maximum number of rounding iterations. If None, will run until convergence.
        :param strategy: rounding strategy to use. Available options are: 'default' (Lovasz algorithm) and 'opt'
        :param z_cut: whether to consider cuts through the ellipsoid's center. Only affects the 'opt' strategy.
        :param sphere_cuts: whether to compute the optimal cut over the sphere. Only affects the 'opt' strategy.
        :param sphere_cuts_optimizer: any function optimizer(x0, func, grad) which computes the minimum of 'func' over the unit sphere.
                                      x0 is the starting point, and 'grad' computes the gradient of func projected over the sphere.
                                      If None, we will use our default algorithm, SphericalGradientDescent.
        """
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)

        self.max_iter = max_iter if max_iter is not None else float('inf')
        self.strategy, self.compute_scale_matrix = self.__get_strategy(strategy, z_cut, sphere_cuts, sphere_cuts_optimizer)

    @staticmethod
    def __get_strategy(strategy: str, z_cut: bool, sphere_cuts: bool, sphere_cuts_optimizer) -> Tuple[Strategy, bool]:
        strategy = strategy.upper()
        if strategy == 'DEFAULT':
            return diagonalization_strategy, True
        if strategy == 'OPT':
            return OptimizedStrategy(z_cut, sphere_cuts, sphere_cuts_optimizer), False
        raise ValueError("Unknown strategy {}. Possible values are: 'default', 'opt'.")

    @metric_logger.log_execution_time('rounding_fit_time', on_duplicates='sum')
    def fit(self, body: BoundedPolyhedralCone, elp: Optional[Ellipsoid] = None) -> Ellipsoid:
        if elp is None:
            elp = Ellipsoid(body.dim, compute_scale_matrix=self.compute_scale_matrix)

        count = 0
        while count < self.max_iter and self.strategy(elp, body):
            count += 1

        metric_logger.log_metric('rounding_iters', count, on_duplicates='append')

        return elp


def diagonalization_strategy(elp: Ellipsoid, body: BoundedPolyhedralCone) -> bool:
    for vector in elp.extremes():
        hyperplane = body.get_separating_oracle(vector)

        if hyperplane is not None and elp.cut(*hyperplane):
            return True

    return False


class OptimizedStrategy:
    def __init__(self, z_cut: bool = True, sphere_cuts: bool = False, sphere_cuts_optimizer = None):
        self.z_cut = z_cut
        self.sphere_cuts = sphere_cuts
        self.optimizer = SphericalGradientDescent() if sphere_cuts_optimizer is None else sphere_cuts_optimizer

    def __call__(self, elp: Ellipsoid, body: BoundedPolyhedralCone) -> bool:
        alpha, hyperplane = self._get_alpha_cut(elp, body)

        if self.z_cut and alpha != 0:
            alpha_z, hyperplane_z = self._get_z_cut(elp)

            if alpha_z > alpha:
                alpha, hyperplane = alpha_z, hyperplane_z

        threshold = 1 / ((elp.dim + 1) * np.sqrt(elp.dim))
        if -alpha >= threshold:
            if not self.sphere_cuts:
                return False

            gamma_sphere, hyperplane_sphere = self._get_best_cut_on_sphere(elp, threshold)

            if gamma_sphere >= threshold:
                return False

            hyperplane = hyperplane_sphere

        elp.cut(*hyperplane)
        return True

    @staticmethod
    def _get_alpha_cut(elp: Ellipsoid, body: BoundedPolyhedralCone) -> Tuple[float, HyperPlane]:
        alphas = elp.compute_alpha(body.A)
        idx_max = np.argmax(alphas)
        return alphas[idx_max], (0, body.A[idx_max])

    @staticmethod
    def _get_z_cut(elp: Ellipsoid) -> Tuple[float, HyperPlane]:
        hyperplane = np.linalg.norm(elp.center), elp.center
        return elp.compute_alpha_single(*hyperplane), hyperplane

    def _get_best_cut_on_sphere(self, elp: Ellipsoid, threshold: float) -> Tuple[float, HyperPlane]:
        func, grad = alpha_on_sphere(elp)
        self.optimizer.fun_threshold = threshold
        result = self.optimizer.optimize(elp.center, func, grad)
        return result.fun, (1, result.x)


class ResultObject:
    def __init__(self, x, fun, grad, iters=0):
        self.x = x
        self.fun = fun
        self.grad = grad
        self.grad_norm = np.linalg.norm(grad)
        self.converged = False
        self.iters = iters


class SphericalGradientDescent:
    def __init__(self, retraction: str = 'sphere', grad_norm_threshold: float = 1e-7, rel_tol: float = 1e-7,
                 max_iter: Optional[int] = None, fun_threshold: Optional[float] = None, step_size: Optional[float] = None):
        assert_positive(grad_norm_threshold, 'grad_norm_threshold')
        assert_positive(rel_tol, 'rel_tol')
        assert_positive_integer(max_iter, 'max_iter', allow_none=True)
        assert_positive(step_size, 'step_size', allow_none=True)

        self.retraction, self.step_optimizer = self.__get_retraction_and_step_optimizer(retraction, step_size)
        self.grad_norm_threshold = grad_norm_threshold
        self.rel_tol = rel_tol
        self.max_iter = max_iter if max_iter else np.inf
        self.fun_threshold = fun_threshold if fun_threshold is not None else -np.inf

    def __get_retraction_and_step_optimizer(self, retraction: str, step_size: Optional[float]):
        retraction = retraction.upper()

        if retraction == 'SPHERE':
            return (
                lambda step, result: np.cos(step) * result.x - (np.sin(step) / result.grad_norm) * result.grad,
                self.__get_step_optimizer(step_size, bounds=(-np.pi, np.pi), method='Bounded')
            )

        if retraction == 'PROJECT':
            return (
                lambda step, result: self.proj(result.x - step * result.grad),
                self.__get_step_optimizer(step_size, method='Brent')
            )

        raise ValueError('Unknown retraction {}'.format(retraction))

    def __get_step_optimizer(self, step_size: Optional[float] = None, **optimizer_params):
        if step_size is not None:
            return lambda func, result: step_size

        return lambda func, result: scipy.optimize.minimize_scalar(lambda step: func(self.retraction(step, result)), **optimizer_params).x

    @staticmethod
    def proj(x: np.ndarray) -> np.ndarray:
        """ Project a point on the sphere """
        return x / np.linalg.norm(x)

    @metric_logger.log_execution_time('sphere_opt_time', 'append')
    def optimize(self, x0: np.ndarray, func: Callable, grad: Callable) -> ResultObject:
        x = self.proj(x0)

        prev_result = ResultObject(x, func(x), grad(x))
        result = self.advance(prev_result, func, grad)

        while not self.converged(result, prev_result):
            prev_result, result = result, self.advance(result, func, grad)

            if result.iters >= self.max_iter or result.fun < self.fun_threshold:
                return result

        result.converged = True
        return result

    def converged(self, result: ResultObject, prev_result: ResultObject) -> bool:
        return result.grad_norm <= self.grad_norm_threshold or (prev_result.fun - result.fun) < self.rel_tol * prev_result.fun

    def advance(self, result: ResultObject, func: Callable, grad: Callable) -> ResultObject:
        step = self.step_optimizer(func, result)
        x = self.retraction(step, result)
        return ResultObject(x, func(x), grad(x), result.iters + 1)


def alpha_on_sphere(elp: Ellipsoid):
    """
    Returns a method computing f(x) and grad f(x), where
            f(x) = (1 - z^T x) / sqrt(x^T P x)

    grad f(x) is already projected over the tangent plane on the sphere.
    """
    L = elp.L * np.sqrt(elp.D).reshape(1, -1)

    def func(x):
        diff = x - elp.center
        u = L.T.dot(x)
        norm = np.linalg.norm(u)

        return x.dot(diff) / norm

    def grad(x):
        diff = x - elp.center
        u = L.T.dot(x)
        norm = np.linalg.norm(u)

        fun = x.dot(diff) / norm

        sq_norm = norm * norm
        return diff / norm - (fun / sq_norm) * L.dot(u)

    return func, grad
