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
from typing import Optional, Dict

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

    def __init__(self, single_chain=True, warmup: int = 100, thin: int = 10, cache_samples: bool = True,
                 rounding: bool = True, rounding_cache: bool = True, rounding_options: Optional[Dict] = None):
        """
        :param single_chain: whether to generate all samples from a single chain or from multiple chains.
        :param warmup: number of initial samples to skip before generating first sample.
        :param thin: number of samples to skip. Only has an effect if single_chain=True.
        :param cache_samples: whether to cache samples between iterations. Can improve performance.
        :param rounding: whether to apply the rounding preprocessing step. Mixing time considerably improves, but so does
        the running time.
        :param rounding_cache: whether to cache the rounding ellipsoid between iterations. Considerably improves running time.
        :param rounding_options: dictionary containing the rounding algorithm configuration. See RoundingAlgorithm class
        for possible values are defaults.
        """
        assert_positive_integer(warmup, 'warmup')
        assert_positive_integer(thin, 'thin')

        if rounding_options is None:
            rounding_options = {}

        self.chain_sampler = single_chain_sampler(warmup, thin) if single_chain else multiple_chain_sampler(warmup)

        self.rounding_algorithm = RoundingAlgorithm(**rounding_options) if rounding else None

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

        x0 = self.__get_starting_point(version_space, elp)
        chain = HitAndRunChain(x0, version_space, rounding_matrix)
        all_samples = self.chain_sampler(chain, n_samples)

        if self.cache:
            self.samples = all_samples

        return all_samples

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


class HitAndRunChain:
    def __init__(self, x0: np.ndarray, convex_body: BoundedPolyhedralCone, rounding_matrix: Optional[np.ndarray] = None):
        self._x0 = x0
        self._convex_body = convex_body
        self._rounding_matrix = rounding_matrix

        self._x = self._x0.copy()  # copy to avoid changing self._x0

    @property
    def dim(self):
        return self._x0.shape[0]

    def reset(self):
        self._x = self._x0.copy()

    def advance(self, steps: int) -> np.ndarray:
        """
        :param steps: total number of steps to advance
        :return: the sample generated by hit-and-run after advancing
        """
        for i in range(steps):
            self._advance_single()

        return self._x.copy()  # return a copy since self._x is modified every advance() call

    def _advance_single(self) -> None:
        direction = self._sample_direction(self._convex_body.dim)

        t1, t2 = self._convex_body.intersection(self._x, direction)

        t_rand = np.random.uniform(t1, t2)
        self._x += t_rand * direction

    def _sample_direction(self, dim: int) -> np.ndarray:
        direction = np.random.normal(size=dim)

        if self._rounding_matrix is not None:
            direction = self._rounding_matrix.dot(direction)

        return direction


def single_chain_sampler(warmup: int, thin: int):
    assert_positive_integer(warmup, 'warmup')
    assert_positive_integer(thin, 'thin')

    @metric_logger.log_execution_time('hit_and_run_time', on_duplicates='sum')
    def sampler(chain: HitAndRunChain, sample_size: int):
        samples = np.empty((sample_size, chain.dim))

        chain.reset()
        samples[0] = chain.advance(warmup)

        for i in range(1, sample_size):
            samples[i] = chain.advance(thin)

        return samples

    return sampler


def multiple_chain_sampler(chain_length: int):
    assert_positive_integer(chain_length, 'chain_length')

    @metric_logger.log_execution_time('hit_and_run_time', on_duplicates='sum')
    def sampler(chain: HitAndRunChain, sample_size: int):
        samples = np.empty((sample_size, chain.dim))

        for i in range(sample_size):  # TODO: parallelize this loop
            chain.reset()
            samples[i] = chain.advance(chain_length)

        return samples

    return sampler
