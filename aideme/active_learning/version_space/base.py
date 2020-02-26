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

from .kernel import KernelLogisticRegression
from .linear import BayesianLogisticRegression
from ..uncertainty import UncertaintySampler


class LinearVersionSpace(UncertaintySampler):
    def __init__(self, sampling: str = 'deterministic', n_samples: int = 8, warmup: int = 100, thin: int = 10, sigma: float = 100,
                 cache: bool = True, rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt', z_cut: bool = False,
                 rounding_cache: bool = True, use_cython: bool = True, add_intercept: bool = True):
        clf = BayesianLogisticRegression(sampling=sampling, n_samples=n_samples, warmup=warmup, thin=thin, sigma=sigma,
                                         cache=cache, rounding=rounding, max_rounding_iters=max_rounding_iters,
                                         strategy=strategy, z_cut=z_cut, rounding_cache=rounding_cache,
                                         use_cython=use_cython, add_intercept=add_intercept)
        UncertaintySampler.__init__(self, clf)

    def clear(self):
        self.clf.clear()


class KernelVersionSpace(UncertaintySampler):
    def __init__(self, sampling: str = 'deterministic', n_samples: int = 8, warmup: int = 100, thin: int = 10, sigma: float = 100,
                 cache: bool = True, rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt', z_cut: bool = False,
                 rounding_cache: bool = True, use_cython: bool = True, add_intercept: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        clf = KernelLogisticRegression(n_samples=n_samples, add_intercept=add_intercept, sampling=sampling,
                                       warmup=warmup, thin=thin, sigma=sigma, cache=cache,
                                       rounding=rounding, max_rounding_iters=max_rounding_iters,
                                       strategy=strategy, z_cut=z_cut, rounding_cache=rounding_cache, use_cython=use_cython,
                                       kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter)
        UncertaintySampler.__init__(self, clf)

    def clear(self):
        self.clf.clear()
