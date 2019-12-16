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
from scipy.special import xlogy, xlog1py

from .active_learner import ActiveLearner
from .query_by_committee.kernel import KernelLogisticRegression

from scipy.special import xlogy, xlog1py


class EntropyReductionLearner(ActiveLearner):

    def __init__(self, background_sample_size=float('inf'), n_samples=8, add_intercept=True, sampling='deterministic',
                 warmup=100, thin=1, sigma=100.0, rounding=True,
                 kernel='rbf', gamma=None, degree=3, coef0=0):
        self.kernel_logreg = KernelLogisticRegression(
            n_samples=n_samples, add_intercept=add_intercept,
            sampling=sampling, warmup=warmup, thin=thin, sigma=sigma, rounding=rounding,
            kernel=kernel, gamma=gamma, degree=degree, coef0=coef0
        )

        self.background_sample_size = background_sample_size

    def fit(self, X, y):
        self.kernel_logreg.fit(X, y)

    def predict(self, X):
        return self.kernel_logreg.predict(X)

    def predict_proba(self, X):
        return self.kernel_logreg.predict_proba(X)

    def rank(self, X):
        H = self.kernel_logreg.sample_predictions(X)  # samples x data points

        row_idx = np.arange(H.shape[0])
        scores = np.zeros(len(X))

        for i, x in enumerate(X):
            mask = H[:, i] > 0

            if np.mean(mask) in [0, 1]:
                scores[i] = float('inf')
                continue

            col_sample = self.__sample_indexes(H.shape[1])

            pos_proba = H[row_idx[mask], col_sample[:, np.newaxis]].mean(axis=0)

            neg_proba = H[row_idx[~mask], col_sample[:, np.newaxis]].mean(axis=0)

            scores[i] = max(self.__compute_average_entropy(pos_proba), self.__compute_average_entropy(neg_proba))

        return scores

    @classmethod
    def __compute_average_entropy(cls, p):
        return -(xlogy(p, p) + xlog1py(1 - p, -p)).mean()

    def __sample_indexes(self, size):
        idx = np.arange(size)

        if self.background_sample_size >= size:
            return size

        return np.random.choice(idx, size=self.background_sample_size, replace=False)