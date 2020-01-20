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

from typing import TYPE_CHECKING

import numpy as np

from .active_learner import ActiveLearner

if TYPE_CHECKING:
    pass


class RandomSampler(ActiveLearner):
    """
    Randomly picks the next point to label. Usually used as baseline method for comparison.
    """
    def __init__(self, clf):
        """
        :param clf: Classifier object implementing two methods:
            - fit(X, y): fits the classifier over the labeled data X,y
            - predict(X): returns the class labels for a given set X

            Additionally, this object can use implement predict_proba(X), but it is not mandatory.
        """
        self._clf = clf

    def fit(self, X, y):
        self._clf.fit(X, y)

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def rank(self, X):
        return np.random.permutation(len(X))
