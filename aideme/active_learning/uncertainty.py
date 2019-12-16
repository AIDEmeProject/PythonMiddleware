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

from .active_learner import ActiveLearner


class UncertaintySampler(ActiveLearner):
    """
    Uncertainty sampling is the most simple and popular Active Learning class of algorithms. Basically, given any classifier
    (Random Forest, SVM, ...) it will rank unlabeled points based one the estimated class probabilities: the closest to 0.5,
    the most uncertain (hence more informative) the sample is.
    """
    def __init__(self, clf):
        """
        :param clf: classifier object implementing 3 methods:
            - fit(X, y): trains a classification model over the labeled data
            - predict(X): predicts class label for each row of matrix X
            - predict_proba(X): predicts probability of being positive for each row of matrix X

            It should be compatible with most Scikit-learn library's classifiers.
        """
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def rank(self, X):
        return np.abs(self.predict_proba(X) - 0.5)
