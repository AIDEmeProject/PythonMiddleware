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


class LabeledSet:
    """
    This class manages a collection of user labels, including final labels, partial labels, and the corresponding data
    point indexes.
    """
    def __init__(self, labels, partial=None, index=None):
        """
        :param labels: the collection of user labels (0, 1 format)
        :param partial: the collection of user partial labels, as a matrix of 0, 1 values. Use None if there is no partial
        labels information from user.
        :param index: indexes corresponding to each label. If None, a range index will be assumed.
        """
        self.labels = np.ravel(labels)
        self.partial = self.__get_partial_labels(partial)
        self.index = self.__get_index(index)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        """
        :param item: a slice of positions to be retrieved
        :return: a LabeledSet instance containing the selected slice
        """
        return LabeledSet(self.labels[item], self.partial[item], self.index[item])

    def __get_partial_labels(self, partial_labels):
        if partial_labels is None:
            return self.labels.reshape(-1, 1)

        partial_labels = np.asarray(partial_labels)

        if partial_labels.ndim != 2:
            raise ValueError("Expected two-dimensional array of partial labels, but ndim = {}".format(partial_labels.ndim))

        if partial_labels.shape[0] != len(self):
            raise ValueError("Wrong size of partial_labels: expected {}, but got {}".format(partial_labels.shape[0], len(self)))

        return partial_labels

    def __get_index(self, idx):
        if idx is None:
            return np.arange(len(self))

        idx = np.ravel(idx)

        if len(idx) != len(self):
            raise ValueError("Wrong size of indexes: expected {}, but got {}".format(len(idx), len(self)))

        return idx

    def asdict(self):
        """
        :return: a dict containing all index and labels information
        """
        return {
            'labeled_indexes': self.index,
            'final_labels': self.labels.tolist(),
            'partial_labels': self.partial.tolist(),
        }
