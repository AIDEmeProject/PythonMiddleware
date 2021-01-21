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
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from sklearn.metrics import f1_score

from aideme.utils import assert_positive_integer
from .linear import LinearFactorizationLearner
from ..active_learner import ActiveLearner

if TYPE_CHECKING:
    from aideme.explore import PartitionedDataset


class SwapLearner(ActiveLearner):
    def __init__(self, active_learner: ActiveLearner, linear_model: LinearFactorizationLearner, refined_model: Optional[LinearFactorizationLearner] = None,
                 swap_iter: int = 100, num_subspaces: int = 15, retries: int = 10, retrain: bool = False, train_sample_size: int = None):
        assert_positive_integer(swap_iter, 'swap_iter')
        assert_positive_integer(num_subspaces, 'num_subspaces')
        assert_positive_integer(retries, 'retries')
        assert_positive_integer(train_sample_size, 'train_sample_size', allow_none=True)

        self._active_learner = active_learner

        self._linear_model = None
        self._swap_model = linear_model
        self._refined_model = linear_model if refined_model is None else refined_model
        self._swap_iter = swap_iter
        self._num_subspaces = num_subspaces
        self._retries = retries
        self._retrain = retrain
        self._train_sample_size = train_sample_size

        self.__it = 0

    def clear(self) -> None:
        self._active_learner.clear()
        self._linear_model = None
        self.__it = 0

    def fit_data(self, data: PartitionedDataset) -> None:
        self.__it += 1

        if self.__it <= self._swap_iter:
            # in early iterations, run AL
            self._active_learner.fit_data(data)
        elif self.__it == self._swap_iter + 1:
            # at the swap iteration, we train a FLM mimicking the AL
            # TODO: always include labeled data in test data?
            X_test = data.sample(self._train_sample_size)
            y_test = self._active_learner.predict(X_test)
            self._swap_model.fit(X_test, y_test, self._num_subspaces, retries=self._retries, x0=None)
            self._linear_model = self._swap_model
            print('pred acc:', f1_score(y_test, self._swap_model.predict(X_test)))
        else:
            # in later iterations, we simply refine the previously computed FLM with the new labeled data
            X, y = data.training_set()
            retries, x0 = (self._retries, None) if self._retrain else (1, self._linear_model.weight_matrix)
            self._refined_model.fit(X, y, self._linear_model.num_subspaces, retries=retries, x0=x0)
            self._linear_model = self._refined_model

        X, y = data.training_set()
        print('it: {}, fscore: {}'.format(self.__it, f1_score(y, self.predict(X))))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.__it <= self._swap_iter:
            return self._active_learner.predict(X)
        else:
            return self._linear_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.__it <= self._swap_iter:
            return self._active_learner.predict_proba(X)
        else:
            return self._linear_model.predict_proba(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        if self.__it <= self._swap_iter:
            return self._active_learner.rank(X)
        else:
            return np.abs(self._linear_model.predict_proba(X) - 0.5)
