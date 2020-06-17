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
import sklearn

from .active_learner import ActiveLearner
from ..utils import assert_positive, assert_positive_integer


class QueryByDisagreement(ActiveLearner):

    def __init__(self, learner, background_sample_size: int = 200, background_sample_weight: float = 1e-5):
        assert_positive_integer(background_sample_size, 'background_sample_size')
        assert_positive(background_sample_weight, 'background_sample_weight')

        self._background_sample_size = background_sample_size
        self._background_sample_weight = background_sample_weight

        self._learner = learner
        self._positively_biased_learner = sklearn.base.clone(learner)
        self._negatively_biased_learner = sklearn.base.clone(learner)

    def fit_data(self, data) -> None:
        X, y = data.training_set()
        self._learner.fit(X, y)

        background_points = data.unlabeled.sample(self._background_sample_size).data
        X_train = np.r_[X, background_points]

        sample_weights = np.ones(len(X_train))
        sample_weights[-self._background_sample_size:] *= self._background_sample_weight

        y_train = np.r_[y, np.ones(self._background_sample_size)]
        self._positively_biased_learner.fit(X_train, y_train, sample_weights)

        y_train[-self._background_sample_size:] = 0
        self._negatively_biased_learner.fit(X_train, y_train, sample_weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._learner.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._learner.predict_proba(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        """ Pick a random point for which the positively and negatively biased classifiers differ. """

        positively_biased_labels = self._positively_biased_learner.predict(X)  # type: np.ndarray
        negatively_biased_labels = self._negatively_biased_learner.predict(X)  # type: np.ndarray

        return (positively_biased_labels == negatively_biased_labels).astype('float')
