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

from numpy import arange
from sklearn.utils import check_random_state, check_array

from .utils import assert_positive_integer


class StratifiedSampler:
    """
        Binary stratified sampling method. Randomly selects a given number of positive and negative points from an array
        of labels.
    """
    def __init__(self, pos, neg, assert_neg_all_subspaces=False, random_state=None):
        """

        :param pos: Number of positive points to sample. Must be non-negative.
        :param neg: Number of negative points to sample. Must be non-negative.
        """
        assert_positive_integer(pos, 'pos')
        assert_positive_integer(neg, 'neg')

        self.pos = pos
        self.neg = neg
        self.assert_neg_all_subspaces = assert_neg_all_subspaces
        self.__random_state = check_random_state(random_state)

    def __call__(self, y, true_class=1, neg_class=0):
        """
        Call the sampling procedure over the input array.

        :param y: array-like collection of labels
        :param true_class: class to be considered positive. Default to 1.0.
        :return: index of samples in the array
        """
        y = check_array(y, ensure_2d=False, allow_nd=False)

        # TODO: can we avoid this check ?
        if y.ndim == 2:
            y = y.mean(axis=1) if self.assert_neg_all_subspaces else y.min(axis=1)

        idx = arange(len(y))
        pos_samples = self.__random_state.choice(idx[y == true_class], size=self.pos, replace=False)
        neg_samples = self.__random_state.choice(idx[y == neg_class], size=self.neg, replace=False)

        return list(pos_samples) + list(neg_samples)


class FixedSampler:
    """
        Dummy sampler which returns a specified selection of indexes.
    """
    def __init__(self, indexes):
        self.indexes = indexes.copy()

    def __call__(self, y):
        return self.indexes
