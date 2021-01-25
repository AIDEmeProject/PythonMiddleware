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

from aideme.utils import assert_positive_integer, assert_in_range
from .learn import prune_irrelevant_subspaces
from .linear import LinearFactorizationLearner
from ..active_learner import ActiveLearner

if TYPE_CHECKING:
    from aideme.explore import PartitionedDataset


class SwapLearner(ActiveLearner):
    def __init__(self, active_learner: ActiveLearner,
                 swap_model: LinearFactorizationLearner, refining_model: Optional[LinearFactorizationLearner] = None, num_subspaces: int = 10, retries: int = 10,
                 swap_iter: int = 100, train_on_prediction: bool = True, train_sample_size: Optional[int] = None,
                 prune: bool = True, prune_threshold: float = 0.99):
        assert_positive_integer(swap_iter, 'swap_iter')
        assert_positive_integer(num_subspaces, 'num_subspaces')
        assert_positive_integer(retries, 'retries')
        assert_positive_integer(train_sample_size, 'train_sample_size', allow_none=True)
        assert_in_range(prune_threshold, 'prune_threshold', 0, 1)

        self._active_learner = active_learner

        self._swap_model = swap_model
        self._refining_model = swap_model if refining_model is None else refining_model
        self._swap_iter = swap_iter
        self._num_subspaces = num_subspaces
        self._retries = retries
        self._train_on_prediction = train_on_prediction
        self._train_sample_size = train_sample_size
        self._prune = prune
        self._prune_threshold = prune_threshold

        self.__it = 0

    def clear(self) -> None:
        self._active_learner.clear()
        self._swap_model.clear()
        self._refining_model.clear()
        self.__it = 0

    @property
    def is_active_learning_phase(self) -> bool:
        return self.__it <= self._swap_iter

    @property
    def is_factorization_phase(self) -> bool:
        return not self.is_active_learning_phase

    @property
    def is_swap_iter(self) -> bool:
        return self.__it == self._swap_iter + 1

    @property
    def linear_model(self) -> Optional[LinearFactorizationLearner]:
        return None if self.is_active_learning_phase else self._refining_model

    def fit_data(self, data: PartitionedDataset) -> None:
        self.__it += 1

        if self.is_active_learning_phase:
            self._active_learner.fit_data(data)

        elif self.is_swap_iter:
            self.__fit_swap_model(data)
            self.__fit_refining_model(data, self._swap_model)

        else:
            self.__fit_refining_model(data, self._refining_model)

    def __fit_swap_model(self, data: PartitionedDataset) -> None:
        if self._train_on_prediction:
            X = data.sample(self._train_sample_size)
            y = self._active_learner.predict(X)
        else:
            X, y = data.training_set()

        self._swap_model.fit(X, y, self._num_subspaces, retries=self._retries, x0=None)

    def __fit_refining_model(self, data: PartitionedDataset, prev_model: LinearFactorizationLearner) -> None:
        X, y = data.training_set()
        self._refining_model.fit(X, y, prev_model.num_subspaces, x0=prev_model.weight_matrix)

        if self._prune:
            self._refining_model = prune_irrelevant_subspaces(data.data, self._refining_model, threshold=self._prune_threshold)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.is_active_learning_phase:
            return self._active_learner.predict(X)
        else:
            return self._refining_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.is_active_learning_phase:
            return self._active_learner.predict_proba(X)
        else:
            return self._refining_model.predict_proba(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        if self.is_active_learning_phase:
            return self._active_learner.rank(X)
        else:
            return np.abs(self._refining_model.predict_proba(X) - 0.5)