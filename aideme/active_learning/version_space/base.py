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
from typing import Union

from .bayesian import ApproximateBayesianLogisticRegression
from .kernel import KernelBayesianLogisticRegression
from .linear import BayesianLogisticRegression, DeterministicLogisticRegression, BayesianLogisticRegressionBase
from ..uncertainty import UncertaintySampler


class VersionSpaceBase(UncertaintySampler):
    def __init__(self, logreg: Union[KernelBayesianLogisticRegression, BayesianLogisticRegressionBase]):
        UncertaintySampler.__init__(self, logreg)

    def clear(self) -> None:
        self.clf.clear()


class LinearVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10,
                 cache: bool = True, rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt', z_cut: bool = False,
                 rounding_cache: bool = True, use_cython: bool = True, add_intercept: bool = True):
        logreg = DeterministicLogisticRegression(
            n_samples=n_samples, warmup=warmup, thin=thin,
            cache=cache, rounding=rounding, max_rounding_iters=max_rounding_iters, strategy=strategy, z_cut=z_cut,
            rounding_cache=rounding_cache,
            use_cython=use_cython, add_intercept=add_intercept
        )

        super().__init__(logreg)


class BayesianLinearVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 sampler: str = 'approximate', prior: str = 'improper', prior_std: float = 1.0, suppress_warnings: bool = True):
        if sampler == 'stan':
            logreg = BayesianLogisticRegression(
                n_samples=n_samples, warmup=warmup, thin=thin, add_intercept=add_intercept,
                prior=prior, prior_std=prior_std, suppress_warnings=suppress_warnings
            )
        elif sampler == 'approximate':
            logreg = ApproximateBayesianLogisticRegression(prior=prior, prior_std=prior_std, add_intercept=add_intercept)
        else:
            raise ValueError("Unknown sampler option: {}".format(sampler))

        super().__init__(logreg)


class KernelVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10,
                 cache: bool = True, rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt', z_cut: bool = False,
                 rounding_cache: bool = True, use_cython: bool = True, add_intercept: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        logreg = DeterministicLogisticRegression(
            n_samples=n_samples, warmup=warmup, thin=thin,
            cache=cache, rounding=rounding, max_rounding_iters=max_rounding_iters, strategy=strategy, z_cut=z_cut, rounding_cache=rounding_cache,
            use_cython=use_cython, add_intercept=add_intercept
        )

        kernel_logreg = KernelBayesianLogisticRegression(
            logreg, decompose=rounding_cache,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        super().__init__(kernel_logreg)


class BayesianKernelVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 sampler: str = 'approximate', prior: str = 'improper', prior_std: float = 1.0, suppress_warnings: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        if sampler == 'stan':
            logreg = BayesianLogisticRegression(
                n_samples=n_samples, warmup=warmup, thin=thin, add_intercept=add_intercept,
                prior=prior, prior_std=prior_std, suppress_warnings=suppress_warnings
            )
        elif sampler == 'approximate':
            logreg = ApproximateBayesianLogisticRegression(prior=prior, prior_std=prior_std, add_intercept=add_intercept)
        else:
            raise ValueError("Unknown sampler option: {}".format(sampler))

        kernel_logreg = KernelBayesianLogisticRegression(
            logreg, decompose=False,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        super().__init__(kernel_logreg)
