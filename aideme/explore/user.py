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

import math
import numpy as np

from ..utils import assert_positive_integer


class User:
    @property
    def is_willing(self):
        raise NotImplementedError

    def clear(self):
        pass

    def label(self, idx, X):
        raise NotImplementedError


class DummyUser(User):
    def __init__(self, final_labels, partial_labels=None, max_iter=math.inf):
        assert_positive_integer(max_iter, 'max_iter', allow_inf=True)

        self.final_labels = np.ravel(final_labels)
        self.partial_labels = self.__get_partial_labels(partial_labels)

        self.__max_iter = max_iter
        self.__num_labeled_points = 0

    def __get_partial_labels(self, partial_labels):
        if partial_labels is None:
            return self.final_labels.reshape(-1, 1)

        partial_labels = np.atleast_2d(partial_labels)
        if len(partial_labels) != len(self.final_labels):
            raise ValueError("Partial and final labels have incompatible lengths: {} != {}".format(len(partial_labels), len(self.final_labels)))
        return partial_labels

    @property
    def is_willing(self):
        return self.__num_labeled_points <= self.__max_iter

    def clear(self):
        self.__num_labeled_points = 0

    def label(self, idx, X):
        self.__num_labeled_points += len(idx)
        return self.partial_labels[idx], self.final_labels[idx]


class CommandLineUser(User):
    @property
    def is_willing(self):
        val = input("Continue (y/n): ")
        while val not in ['y', 'n']:
            val = input("Continue (y/n): ")

        return True if val == 'y' else False

    def label(self, idx, pts):
        is_valid, labels = self.__is_valid_input(pts)
        while not is_valid:
            is_valid, labels = self.__is_valid_input(pts)
        return labels

    @staticmethod
    def __is_valid_input(pts):
        s = input("Give the labels for the following points: {}\n".format(pts.tolist()))
        expected_size = len(pts)

        if not set(s).issubset({' ', '0', '1'}):
            print("Invalid character in labels. Only 0, 1 and ' ' are permitted.\n")
            return False, None

        vals = s.split()
        if len(vals) != expected_size:
            print('Incorrect number of labels: got {} but expected {}\n'.format(len(vals), expected_size))
            return False, None

        print()
        return True, [int(x) for x in vals]

