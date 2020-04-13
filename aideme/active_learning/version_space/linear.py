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

from .sampling import StanLogisticRegressionSampler, HitAndRunSampler


class BayesianLogisticRegressionBase:
    def __init__(self, sampler,  n_samples: int = 8, add_intercept: bool = True, intercept_value: float = 1.):
        """
        :param sampler: sampling method
        :param n_samples: number of samples to compute from posterior
        :param add_intercept: whether to add an intercept or not
        :param intercept_value: value in column appended to data matrix during fit
        """
        self.sampler = sampler
        self.n_samples = n_samples
        self.add_intercept = add_intercept
        self.intercept_value = intercept_value

    def clear(self) -> None:
        self.sampler.clear()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.add_intercept:
            intercept = np.full(shape=(len(X), 1), fill_value=self.intercept_value)
            X = np.hstack([X, intercept])

        samples = self.sampler.sample(X, y, self.n_samples)

        if self.add_intercept:
            self.bias, self.weight = samples[:, -1].reshape(-1, 1), samples[:, :-1]
        else:
            self.bias, self.weight = 0, samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean(self._likelihood(X), axis=0)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _margin(self, X: np.ndarray) -> np.ndarray:
        return self.bias + self.weight.dot(X.T)


class DeterministicLogisticRegression(BayesianLogisticRegressionBase):
    """
    DETERMINISTIC POSTERIOR
        p(w | X, y) = 1 if y_i X_i^T w > 0 for all i, else 0

    In this particular case, we assume no labeling noise, and data should be linear separable. However, we can achieve
    better performance under these assumptions.
    """

    def __init__(self, n_samples: int = 8, warmup: int = 100, thin: int = 10,
                 cache_samples: bool = True, rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt',
                 z_cut: bool = False, rounding_cache: bool = True,
                 add_intercept: bool = True, intercept_value: float = 1.):
        """
        :param n_samples: number of samples to compute from posterior
        :param warmup: number of samples to ignore (MCMC throwaway initial samples)
        :param thin: how many iterations to skip between samples
        :param cache_samples: whether to cache previous samples in order to speed-up 'initial point' computation in hit-and-run.
        :param rounding: whether to apply a rounding procedure in the 'deterministic' sampling.
        :param max_rounding_iters: maximum number of iterations for rounding algorithm
        :param strategy: rounding strategy. Available values are: 'default' and 'opt'
        :param rounding_cache: whether cache rounding ellipsoid between iterations. Significantly speeds-up computations, but performance may suffer a little.
        :param add_intercept: whether to add an intercept or not
        """

        sampler = HitAndRunSampler(warmup=warmup, thin=thin, cache_samples=cache_samples,
                                   rounding=rounding, max_rounding_iters=max_rounding_iters,
                                   strategy=strategy, z_cut=z_cut, rounding_cache=rounding_cache)

        super().__init__(sampler=sampler, n_samples=n_samples, add_intercept=add_intercept, intercept_value=intercept_value)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        return (self._margin(X) > 0).astype('float')


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
                                                suppress_warnings=suppress_warnings,)

        super().__init__(sampler=sampler, n_samples=n_samples, add_intercept=add_intercept)

    def _likelihood(self, X: np.ndarray) -> np.ndarray:
        return expit(self._margin(X))
