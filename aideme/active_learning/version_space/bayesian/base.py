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
from .linear import StanBayesianLogisticRegression
from .laplace import LaplaceBayesianLogisticRegression, KernelLaplaceBayesianLogisticRegression
from ..base import VersionSpaceBase
from ..kernel import KernelBayesianLogisticRegression


class BayesianLinearVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 sampler: str = 'laplace', prior: str = 'improper', prior_std: float = 1.0,
                 tol: float = 1e-6, max_iter: int = 10000, suppress_warnings: bool = True):
        if sampler == 'stan':
            logreg = StanBayesianLogisticRegression(
                n_samples=n_samples, warmup=warmup, thin=thin, add_intercept=add_intercept,
                prior=prior, prior_std=prior_std, suppress_warnings=suppress_warnings
            )
        elif sampler == 'laplace':
            logreg = LaplaceBayesianLogisticRegression(prior=prior, prior_std=prior_std, add_intercept=add_intercept, tol=tol, max_iter=max_iter)
        else:
            raise ValueError("Unknown sampler option: {}".format(sampler))

        super().__init__(logreg)


class BayesianKernelVersionSpace(VersionSpaceBase):
    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 sampler: str = 'laplace', prior: str = 'improper', prior_std: float = 1.0,
                 tol: float = 1e-6, max_iter: int = 10000, suppress_warnings: bool = True,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):
        if sampler == 'stan':
            logreg = StanBayesianLogisticRegression(
                n_samples=n_samples, warmup=warmup, thin=thin, add_intercept=add_intercept,
                prior=prior, prior_std=prior_std, suppress_warnings=suppress_warnings
            )
        elif sampler == 'laplace':
            logreg = LaplaceBayesianLogisticRegression(prior=prior, prior_std=prior_std, add_intercept=add_intercept, tol=tol, max_iter=max_iter)
        elif sampler == 'kernel-laplace':
            logreg = KernelLaplaceBayesianLogisticRegression(prior_std=prior_std, tol=tol, max_iter=max_iter)
        else:
            raise ValueError("Unknown sampler option: {}. Available options are: 'stan', 'laplace', and 'kernel-laplace'.".format(sampler))

        kernel_logreg = KernelBayesianLogisticRegression(
            logreg, decompose=False,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        super().__init__(kernel_logreg)
