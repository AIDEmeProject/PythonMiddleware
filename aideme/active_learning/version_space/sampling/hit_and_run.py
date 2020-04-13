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
from typing import Optional

import numpy as np

from aideme.utils import assert_positive_integer, metric_logger
from .ellipsoid import Ellipsoid
from .polyhedral_cone import BoundedPolyhedralCone
from .rounding import RoundingAlgorithm


class HitAndRunSampler:
    """
    Hit-and-run is a MCMC sampling technique for sampling uniformly from a Convex Polytope. In our case, we restrict
    this technique for sampling from the Linear Version Space convex body defined above.

    Reference: https://link.springer.com/content/pdf/10.1007%2Fs101070050099.pdf
    """

    def __init__(self, warmup: int = 100, thin: int = 10, cache_samples: bool = True,
                 rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt', z_cut: bool = False,
                 rounding_cache: bool = True):
        """
        :param warmup: number of initial samples to ignore
        :param thin: number of samples to skip
        :param rounding: whether to apply the rounding preprocessing step. Mixing time considerably improves, but so does
        :param max_rounding_iters: maximum number of iterations of rounding algorithm
        :param cache_samples: whether to cache samples between iterations
        the running time.
        """
        assert_positive_integer(warmup, 'warmup')
        assert_positive_integer(thin, 'thin')

        self.warmup = warmup
        self.thin = thin

        self.rounding_algorithm = None
        self.rounding_algorithm = RoundingAlgorithm(max_rounding_iters, strategy=strategy, z_cut=z_cut) if rounding else None

        self.rounding_cache = rounding_cache if rounding else False
        self.ellipsoid_cache = None  # type: Optional[Ellipsoid]

        self.cache = cache_samples
        self.samples = np.array([])

    def clear(self):
        self.ellipsoid_cache = None
        self.samples = np.array([])

    def sample(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Compute a MCMC Sample from the version space.

        :param X: data matrix
        :param y: labels (positive label should be 1)
        :param n_samples: number of samples
        :return: samples in a numpy array (one per line)
        """
        A = X * np.where(y == 1, -1, 1).reshape(-1, 1)
        version_space = BoundedPolyhedralCone(A)

        # rounding
        elp, rounding_matrix = None, None
        if self.rounding_algorithm:
            elp = self.rounding_algorithm.fit(version_space, self.__compute_new_ellipsoid(version_space.dim))

            if self.rounding_cache:
                self.ellipsoid_cache = elp

            rounding_matrix = elp.L * np.sqrt(elp.D).reshape(1, -1)

        all_samples = self.__run_sampling_procedure(n_samples, elp, version_space, rounding_matrix)
        metric_logger.log_metric('hit_and_run_steps', self.warmup + self.thin * (n_samples - 1))

        # cache samples
        if self.cache:
            self.samples = all_samples

        return all_samples

    @metric_logger.log_execution_time('hit_and_run_time', on_duplicates='sum')
    def __run_sampling_procedure(self, n_samples: int, elp: Ellipsoid, version_space: BoundedPolyhedralCone, rounding_matrix: Optional[np.ndarray]):
        cur_sample = self.__get_starting_point(version_space, elp)
        all_samples = np.empty((n_samples, version_space.dim))

        # skip samples
        self.__advance(self.warmup, cur_sample, rounding_matrix, version_space)
        all_samples[0] = cur_sample.copy()

        # thin samples
        for i in range(1, n_samples):
            self.__advance(self.thin, cur_sample, rounding_matrix, version_space)
            all_samples[i] = cur_sample.copy()
        return all_samples

    def __advance(self, n_iter: int, center: np.ndarray, rounding_matrix: Optional[np.ndarray], version_space: BoundedPolyhedralCone) -> None:
        for _ in range(n_iter):
            # sample random direction
            direction = self.__sample_direction(version_space.dim, rounding_matrix)

            # get extremes of line segment determined by intersection
            t1, t2 = version_space.intersection(center, direction)

            # get random point on line segment
            t_rand = np.random.uniform(t1, t2)
            center += t_rand * direction

    @staticmethod
    def __sample_direction(dim: int, rounding_matrix: Optional[np.ndarray]) -> np.ndarray:
        direction = np.random.normal(size=dim)
        return rounding_matrix.dot(direction) if rounding_matrix is not None else direction

    def __get_starting_point(self, version_space: BoundedPolyhedralCone, ellipsoid: Optional[Ellipsoid]) -> np.ndarray:
        if ellipsoid and version_space.is_inside_polytope(ellipsoid.center):
            return ellipsoid.center / np.linalg.norm(ellipsoid.center)

        if self.cache and self.samples.shape[0] > 0:
            samples = self.__truncate_samples(version_space.dim)
            for sample in samples:
                if version_space.is_inside_polytope(sample):
                    return sample.copy()

        print('Falling back to linprog in Hit-and-Run sampler.')  # TODO: log this / raise warning
        return version_space.get_interior_point()

    def __truncate_samples(self, new_dim: int) -> np.ndarray:
        N, dim = self.samples.shape

        if new_dim <= dim:
            return self.samples[:, :new_dim]

        return np.hstack([self.samples, np.zeros((N, new_dim - dim))])

    def __compute_new_ellipsoid(self, new_dim: int) -> Optional[Ellipsoid]:
        if self.ellipsoid_cache is None:  # no rounding
            return None

        d = self.ellipsoid_cache.dim

        if new_dim == d:  # linear case: dimension does not grow
            return self.ellipsoid_cache

        if new_dim != d + 1:
            # TODO: is it possible to perform more general updates?
            raise RuntimeError("Only +1 updates are supported.")

        # kernel case: dimension grows with number of labeled points
        elp = Ellipsoid(new_dim, compute_scale_matrix=False)

        elp.center[:-1] = self.ellipsoid_cache.center

        elp.D[:-1] = self.ellipsoid_cache.D * (1 + 1/d)
        elp.D[-1] = 1 + d

        elp.L[:-1, :-1] = self.ellipsoid_cache.L

        if self.ellipsoid_cache.scale is not None:
            elp.scale = np.zeros(shape=(new_dim, new_dim))
            elp.scale[:-1, :-1] = self.ellipsoid_cache.scale * (1 + 1/d)
            elp.scale[-1, -1] = 1 + d

        return elp
