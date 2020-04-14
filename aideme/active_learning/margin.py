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

"""
SVM-based Active Learning algorithms, from the article "Support Vector Machine Active Learning with Applications to Text
Classification", JMLR (2001), by Simon TONG & Daphne KOLLER.

Link: http://www.jmlr.org/papers/volume2/tong01a/tong01a.pdf
"""

from typing import Union

import numpy as np
from scipy.special import expit
from sklearn import clone
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .uncertainty import UncertaintySampler


class SimpleMargin(UncertaintySampler):
    """
    At every iteration, it trains an SVM model over labeled data, and picks the closest point to the decision boundary
    as most informative point.
    """
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: Union[str, float]='auto'):
        clf = SVC(C=C, kernel=kernel, gamma=gamma, decision_function_shape='ovo')
        super().__init__(clf)

    def predict_proba(self, X):
        return expit(self.clf.decision_function(X))

    def rank(self, X):
        """
        Rank points based on their distance to the decision boundary.
        :param X: data matrix
        :return: distance list
        """
        return np.abs(self.clf.decision_function(X))


class RatioMargin(SimpleMargin):
    """
    For every unlabeled point, we train two SVM models: one positively biased and one negatively biased. The SVM's margin
    is an estimate for the Version Space remaining volume; thus, the point that most closely halves the current version
    space is the one whose two previously computed margins are the closest to each other.
    """
    def fit(self, X, y):
        self.clf.fit(X, y)

        # store training data
        self.X_train = np.vstack([X, np.zeros(X.shape[1])])
        self.y_train = np.hstack([y, 0])

    def rank(self, X):
        # check model is fitted
        check_is_fitted(self.clf, 'support_')

        # clone fitted model to avoid losing its weights
        clf = clone(self.clf)

        # add "-" sign because we want the LARGEST margin to be returned
        return -np.array([self.__compute_margin_ratio(clf, x) for x in X])

    def __compute_margin_ratio(self, clf, x):
        margin0 = self.__compute_margin(clf, x, clf.classes_[0])
        margin1 = self.__compute_margin(clf, x, clf.classes_[1])

        if margin0 <= 0 or margin1 <= 0:
            return float('inf')

        return min(margin1 / margin0, margin0 / margin1)

    def __compute_margin(self, clf, x, y):
        # set training data
        self.X_train[-1] = x
        self.y_train[-1] = y

        # train classifier and
        clf.fit(self.X_train, self.y_train)
        return float(clf.dual_coef_.dot(clf.decision_function(clf.support_vectors_)))
