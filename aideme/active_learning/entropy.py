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
from scipy.special import xlogy

from .active_learner import ActiveLearner
from .version_space.kernel import KernelBayesianLogisticRegression
from .version_space.linear import DeterministicLogisticRegression
from ..utils import assert_positive_integer, metric_logger


class EntropyReductionLearner(ActiveLearner):

    def __init__(self, data_sample_size: Optional[int] = None,
                 single_chain=True, n_samples: int = 8, warmup: int = 100, thin: int = 10, cache_samples: bool = True,
                 rounding: bool = True, rounding_cache: bool = True, rounding_options: Optional[Dict] = None,
                 add_intercept: bool = True, decompose: bool = False,
                 kernel: str = 'rbf', gamma: float = None, degree: int = 3, coef0: float = 0., jitter: float = 1e-12):

        assert_positive_integer(data_sample_size, 'data_sample_size', allow_none=True)

        logreg = DeterministicLogisticRegression(
            single_chain=single_chain, n_samples=n_samples, warmup=warmup, thin=thin, cache_samples=cache_samples,
            rounding=rounding, rounding_cache=rounding_cache, rounding_options=rounding_options,
            add_intercept=add_intercept
        )

        self.kernel_logreg = KernelBayesianLogisticRegression(
            logreg, decompose=decompose,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, jitter=jitter
        )

        self.data_sample_size = data_sample_size
        self.hypothesis_sample_size = n_samples
        self.__sample = None

    def clear(self) -> None:
        self.kernel_logreg.clear()
        self.__sample = None

    def fit_data(self, data) -> None:
        X, y = data.training_set()
        self.kernel_logreg.fit(X, y)

        self.__sample = data.sample(self.data_sample_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kernel_logreg.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.kernel_logreg.predict_proba(X)

    def rank(self, X):
        cut_proba = self.predict_proba(X)
        ranks = np.abs(cut_proba - 0.5)

        min_rank = ranks.min()
        metric_logger.log_metric('min_cut_proba', min_rank)

        if min_rank == 0.5:  # no point cuts the VS
            return np.zeros(len(X))

        # among the points reducing the VS by the most, select the one with smallest expected prediction entropy after labeling
        is_rank_minimizer = (ranks == min_rank)
        ranks[~is_rank_minimizer] = np.inf
        ranks[is_rank_minimizer] = self.__entropy(X[is_rank_minimizer])

        return ranks

    @metric_logger.log_execution_time('entropy_time')
    def __entropy(self, X):
        H = self.kernel_logreg.predict_all(X)  # samples x data points
        H_sample = self.kernel_logreg.predict_all(self.__sample)

        sums = H.sum(axis=0)
        cut_proba = sums / H.shape[0]

        pos_probas = H.T @ H_sample
        neg_probas = H_sample.sum(axis=0) - pos_probas

        pos_probas /= sums.reshape(-1, 1)
        neg_probas /= (H.shape[0] - sums).reshape(-1, 1)

        pos_entropy = self.__compute_average_entropy(pos_probas)
        neg_entropy = self.__compute_average_entropy(neg_probas)

        return pos_entropy * cut_proba + neg_entropy * (1 - cut_proba)

    @staticmethod
    def __compute_average_entropy(p):
        mp = 1 - p
        res = -xlogy(p, p).mean(axis=-1)
        res -= xlogy(mp, mp).mean(axis=-1)
        return res
