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
import numpy as np
from scipy.special import expit

from .stan import StanLogisticRegressionSampler
from ..linear import BayesianLogisticRegressionBase


class StanBayesianLogisticRegression(BayesianLogisticRegressionBase):
    """
    LOGISTIC POSTERIOR
        p(w | X, y) ~= exp(-|w|^2 / sigma^2) / \prod_i (1 + exp(-y_i X_i^T w))

    Basically, we have chosen a centered gaussian prior and a logistic likelihood function.
    """

    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10, add_intercept: bool = True,
                 prior: str = 'improper', prior_std: float = 1.0, suppress_warnings: bool = True):
        """
        :param n_samples: number of samples to compute from posterior
        :param warmup: number of samples to ignore (MCMC throwaway initial samples)
        :param thin: how many iterations to skip between samples
        :param add_intercept: whether to add an intercept or not
        :param prior: prior for logistic regression weights. Available options are: 'gaussian', 'cauchy', and 'improper'
        :param prior_std: standard deviation of prior distribution. It has no effect for 'improper' prior.
        :param suppress_warnings: whether to suppress all pystan warning log messages
        """
        sampler = StanLogisticRegressionSampler(warmup=warmup, thin=thin, prior=prior, prior_std=prior_std,
                                                suppress_warnings=suppress_warnings, )

        super().__init__(sampler=sampler, n_samples=n_samples, add_intercept=add_intercept)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        return expit(self._margin(X))
