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

from ..utils import assert_positive_integer


class User:
    @property
    def is_willing(self):
        raise NotImplementedError

    def label(self, idx, X):
        raise NotImplementedError


class DummyUser(User):
    def __init__(self, labels, max_iter):
        assert_positive_integer(max_iter, 'max_iter', allow_inf=True)

        self.labels = labels
        self.__max_iter = max_iter
        self.__num_labeled_points = 0

    def __repr__(self):
        return 'User num_labeled_points={0} max_iter={1}'.format(self.__num_labeled_points, self.__max_iter)

    @property
    def is_willing(self):
        return self.__num_labeled_points <= self.__max_iter

    def label(self, idx, X):
        self.__num_labeled_points += len(idx)
        return self.labels[idx]
