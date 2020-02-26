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

from ..active_learner import ActiveLearner
from ..dsm.polytope import CategoricalPolytope, MultiSetPolytope


class PolytopeLearner(ActiveLearner):
    """
    Special AL where predictions are simply based over a polytope model instance.
    """
    def __init__(self, pol):
        self.pol = pol

    def clear(self):
        """
        Clears polytope model
        """
        self.pol.clear()

    def fit_data(self, data):
        """
        Updates the polytope model with the last user labeled data.
        :param data: PartitionedDataset instance
        """
        X_new, y_new = data.last_labeled_set()
        self.__update_polytope(X_new, y_new)

    def fit(self, X, y):
        """
        Similar to fit_data, but polytope model is cleared before fitting.
        :param X: data matrix
        :param y: labels
        """
        self.clear()
        self.__update_polytope(X, y)

    def __update_polytope(self, X_new, y_new):
        self.pol.update(X_new, y_new)

        if not self.pol.is_valid:
            raise RuntimeError('Found conflicting labels in polytope: {}'.format(self.pol))

    def predict(self, X):
        """
        Returns the most probable label. 0.5 probabilities are treated as negative labels.
        :param X: data matrix to compute labels
        :return: numpy array containing labels for each data point
        """
        return (self.predict_proba(X) > 0.5).astype('float')

    def predict_proba(self, X):
        """
        :param X: data matrix
        :return: polytope predictions for eac data point x in X
        """
        return self.pol.predict(X)

    def rank(self, X):
        """
        :param X: data matrix
        :return: 0 for points in positive or negative regions, 0.5 otherwise
        """
        return np.abs(self.predict_proba(X) - 0.5)


class CategoricalActiveLearner(PolytopeLearner):
    """
    Special AL for the case where all attributes are categorical. It simply memorizes the positive and negative values
    seen so far.
    """
    def __init__(self):
        super().__init__(CategoricalPolytope())


class MultiSetActiveLearner(PolytopeLearner):
    """
    Special AL for the case where all attributes are come from a multi-set feature. It simply memorizes the positive
    values seen so far, and negative points are cached until there is only one element left, which will assumed to be
    negative.
    """

    def __init__(self):
        super().__init__(MultiSetPolytope())
        self.__n_samples = 50

    def predict_proba(self, X):
        S = np.empty((self.__n_samples, X.shape[1]))

        i = 0
        while i < self.__n_samples:
            new_samples = self.__sample(self.__n_samples - i, X.shape[1])
            new_i = i + len(new_samples)
            S[i:new_i] = new_samples
            i = new_i

        return np.mean(X.dot(S.T) == X.sum(axis=1).reshape(-1, 1), axis=1)

    def __sample(self, n_samples, dim):
        S = np.random.randint(0, 2, size=(n_samples, dim))

        S[:, list(self.pol._pos_indexes)] = 1.0
        S[:, list(self.pol._neg_indexes)] = 0.0

        is_valid = np.full(n_samples, fill_value=True)
        for idx in self.pol._neg_point_cache:
            flags = S[:, list(idx)].sum(axis=1) < len(idx)
            np.logical_and(is_valid, flags, out=is_valid)

        return S[is_valid, :]
