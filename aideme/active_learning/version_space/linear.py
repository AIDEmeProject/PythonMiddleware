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


class BayesianLogisticRegression:
    """
        Logistic regression bayesian classifier. Given a collection of labeled data (X_i, y_i), we assume two choices of posterior:

        LOGISTIC POSTERIOR
            p(w | X, y) ~= exp(-|w|^2 / sigma^2) * \prod_i 1 / (1 + exp(-y_i X_i^T w))

        Basically, we have chosen a centered gaussian prior and a logistic likelihood function.

        DETERMINISTIC POSTERIOR
            p(w | X, y) = 1 if y_i X_i^T w > 0 for all i, else 0

        In this particular case, there we assume no labeling noise, and data should be linear separable. However, we can achieve
        better performance under these assumptions.
    """

    def __init__(self, sampling: str = 'deterministic', n_samples: int = 8, warmup: int = 100, thin: int = 10, sigma: float = 100,
                 cache: bool = True, rounding: bool = True, max_rounding_iters: bool = None, strategy: str = 'opt', z_cut: bool = False,
                 rounding_cache: bool = True, use_cython: bool = True, add_intercept: bool = True):
        """
        :param sampling: sampling method. Options: 'bayesian' (allows labeling noise) and 'deterministic' (no noise)
        :param n_samples: number of samples to compute from posterior
        :param warmup: number of samples to ignore (MCMC throwaway initial samples)
        :param thin: how many iterations to skip between samples
        :param sigma: gaussian prior standard deviation. Works as a L2 regularization (the lower sigma is, the more regularization)
        :param cache: whether to cache previous samples in order to speed-up 'initial point' computation in hit-and-run.
        :param rounding: whether to apply a rounding procedure in the 'deterministic' sampling.
        :param max_rounding_iters: maximum number of iterations for rounding algorithm
        :param strategy: rounding strategy. Available values are: 'default' and 'opt'
        :param rounding_cache: whether cache rounding ellipsoid between iterations. Significantly speeds-up computations, but performance may suffer a little.
        :param add_intercept: whether to add an intercept or not
        """
        if sampling == 'bayesian':
            self.sampler = StanLogisticRegressionSampler(warmup=warmup, thin=thin, sigma=sigma)
        elif sampling == 'deterministic':
            self.sampler = HitAndRunSampler(warmup=warmup, thin=thin, cache=cache,
                                            rounding=rounding, max_rounding_iters=max_rounding_iters,
                                            strategy=strategy, z_cut=z_cut, rounding_cache=rounding_cache,
                                            use_cython=use_cython)
        else:
            raise ValueError("Unknown sampling option. Options are 'deterministic' and 'bayesian'.")

        self.n_samples = n_samples
        self.add_intercept = add_intercept
        self.sampling = sampling

    def clear(self):
        self.sampler.clear()

    def fit(self, X, y):
        # add intercept if needed
        if self.add_intercept:
            ones = np.ones(shape=(len(X), 1))
            X = np.hstack([ones, X])

        # call sampling function
        samples = self.sampler.sample(X, y, self.n_samples)

        # set bias and weight
        if self.add_intercept:
            self.bias, self.weight = samples[:, 0].reshape(-1, 1), samples[:, 1:]
        else:
            self.bias, self.weight = 0, samples

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X):
        return np.mean(self.__likelihood(X), axis=0)

    def __likelihood(self, X):
        margin = self.__margin(X)
        if self.sampling == 'bayesian':
            return expit(margin)
        else:
            return (margin > 0).astype('float')

    def __margin(self, X):
        return self.bias + self.weight.dot(X.T)
