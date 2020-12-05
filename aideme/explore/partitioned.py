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

from typing import Iterable, Tuple, Optional

import numpy as np
import sklearn.utils

from . import LabeledSet
from .index import Index


class PartitionedDataset:  # TODO: how can we add partition information? (factorization)
    """
    The purpose of this class is to store a partitioning of data into 3 parts:
        - LABELED: data points labeled by user, together with their labels
        - INFERRED: data points whose true label has been inferred by the AL algorithm
        - UNKNOWN: remaining data points, whose labels are still unknown

    This class allows for efficient access to data in these three partitions, avoiding data copying as much as possible.
    """

    def __init__(self, data: np.ndarray, index: Optional[np.ndarray] = None, copy=True):
        """

        :param data: data matrix
        :param index: optional data indexes
        :param copy: whether to store a copy of both data and indexes. The partitioning will be done in-place, meaning
        it will change the original data and index arrays when no copy is performed.
        """
        self.__dataset = IndexedDataset(data, index, copy=copy, check_data=True)
        self.__index_to_row = Index(self.__dataset.index)
        self.clear()

    def __len__(self):
        """
        :return: size of original dataset
        """
        return len(self.__dataset)

    def __repr__(self):
        return self.__dataset.__repr__()

    def clear(self) -> None:
        """
        Resets internal data structures, putting all data in the unknown partition. Data is also re-ordered, so it is
        sorted by index.
        """
        self.__labeled_set = LabeledSet([])
        self.__inferred_start = 0
        self.__unknown_start = 0
        self.__last_user_labeled_set = LabeledSet([])

        for i, idx in enumerate(np.sort(self.__dataset.index)):
            self.__swap_rows(i, self.__index_to_row[idx])

    ##################
    # MOVING
    ##################
    def move_to_labeled(self, labeled_set: LabeledSet) -> None:
        """
        Move indexes to LABELED partition, and store its labels. Throws exception if any index is already in this partition.
        :param labeled_set: LabeledSet instance containing the labels and indexes of points to be moved
        """
        for idx in labeled_set.index:
            pos = self.__index_to_row[idx]

            if pos >= self.__inferred_start:
                self.__move_left(pos, to_labeled=True)
            else:
                self.__raise_error(idx, pos, 'labeled')

        self.__last_user_labeled_set = labeled_set
        self.__labeled_set = self.__labeled_set.concat(labeled_set)

    def move_to_inferred(self, indexes: Iterable) -> None:
        """
        Move indexes to INFERRED partition. Throws exception if any index is already in this partition.
        """
        for idx in indexes:
            pos = self.__index_to_row[idx]

            if pos >= self.__unknown_start:
                self.__move_left(pos, to_labeled=False)
            elif pos < self.__inferred_start:
                self.__move_right(pos, to_unknown=False)
            else:
                self.__raise_error(idx, pos, 'inferred')

    def move_to_unknown(self, indexes: Iterable) -> None:
        """
        Move indexes to UNKNOWN partition. Throws exception if any index is already in this partition.
        """
        for idx in indexes:
            pos = self.__index_to_row[idx]

            if pos < self.__unknown_start:
                self.__move_right(idx, to_unknown=True)
            else:
                self.__raise_error(idx, pos, 'unknown')

    def __move_left(self, pos: int, to_labeled: bool) -> None:
        if pos >= self.__unknown_start:
            self.__swap_rows(pos, self.__unknown_start)
            pos = self.__unknown_start
            self.__unknown_start += 1

        if to_labeled and pos >= self.__inferred_start:
            self.__swap_rows(pos, self.__inferred_start)
            self.__inferred_start += 1

    def __move_right(self, pos: int, to_unknown: bool) -> None:
        if pos < self.__inferred_start:
            self.__swap_rows(pos, self.__inferred_start - 1)
            pos = self.__inferred_start - 1
            self.__inferred_start -= 1

        if to_unknown and pos < self.__unknown_start:
            self.__swap_rows(pos, self.__unknown_start - 1)
            self.__unknown_start -= 1

    def __swap_rows(self, i: int, j: int) -> None:
        idx_i, idx_j = self.__dataset.index[i], self.__dataset.index[j]
        self.__dataset.swap_rows(i, j)
        self.__index_to_row.swap_index(idx_i, idx_j)

    def __raise_error(self, idx, pos: int, partition_name: str) -> None:
        raise ValueError("Index {}, at position {}, is already in {} set.".format(idx, pos, partition_name))

    def remove_inferred(self) -> None:
        """
        Moves all points inside the INFERRED partition to the UNKNOWN one.
        """
        self.__unknown_start = self.__inferred_start  # flush inferred partition

    ##################
    # SIZES
    ##################
    @property
    def labeled_size(self) -> int:
        """
        :return: number of points in the LABELED partition
        """
        return self.__inferred_start

    @property
    def inferred_size(self) -> int:
        """
        :return: number of points in the INFERRED partition
        """
        return self.__unknown_start - self.__inferred_start

    @property
    def unknown_size(self) -> int:
        """
        :return: number of points in the UNKNOWN partition
        """
        return len(self) - self.__unknown_start

    @property
    def unlabeled_size(self) -> int:
        """
        :return: number of unlabeled points (INFERRED + UNKNOWN)
        """
        return len(self) - self.__inferred_start

    ##################
    # DATA ACCESS
    ##################
    @property
    def index(self) -> np.ndarray:
        """
        :return: dataset indexes (they will be modified after every 'moving' operation)
        """
        return self.__dataset.index

    @property
    def data(self) -> np.ndarray:
        """
        :return: data matrix (it will be modified after every 'moving' operation)
        """
        return self.__dataset.data

    @property
    def labeled(self) -> IndexedDataset:
        """
        :return: all points (data + index) in the LABELED partition
        """
        return self.__dataset[:self.__inferred_start]

    @property
    def inferred(self) -> IndexedDataset:
        """
        :return: all points (data + index) in the INFERRED partition
        """
        return self.__dataset[self.__inferred_start:self.__unknown_start]

    @property
    def unknown(self) -> IndexedDataset:
        """
        :return: all points (data + index) in the UNKNOWN partition
        """
        return self.__dataset[self.__unknown_start:]

    @property
    def unlabeled(self) -> IndexedDataset:
        """
        :return: all unlabeled points (INFERRED + UNKNOWN)
        """
        return self.__dataset[self.__inferred_start:]

    def sample(self, size: Optional[int] = None) -> np.ndarray:
        """
        :return: a sample without replacement of given size from the entire data
        """
        return self.__dataset.sample(size).data

    ##################
    # LABELED DATA
    ##################
    def last_training_set(self, get_partial=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param get_partial: whether to get the partial labels
        :return: a pair (X, y) of the last points moved into the LABELED partition. X is a data matrix and y a labels array.
        """
        rows = self.__index_to_row.get_rows(self.__last_user_labeled_set.index)
        return self.__dataset.data[rows], self.__last_user_labeled_set.partial if get_partial else self.__last_user_labeled_set.labels

    def training_set(self, get_partial=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param get_partial: whether to get the partial labels
        :return: a pair (X, y) of all points in the LABELED partition (i.e. moved to the LABELED partition). X is a data matrix and y a labels array.
        """
        return self.labeled.data, self.__labeled_set.partial if get_partial else self.__labeled_set.labels

    @property
    def labeled_set(self) -> LabeledSet:
        """
        :return: a LabeledSet instance containing indexes and labels of points in the LABELED partition
        """
        return self.__labeled_set


class IndexedDataset:
    def __init__(self, data: np.ndarray, index: Optional[np.ndarray] = None, copy: bool = False, check_data: bool = True):
        if index is None:
            index = np.arange(data.shape[0])

        if check_data:
            data, index = sklearn.utils.check_X_y(data, index, ensure_min_samples=0)

        if copy:
            data, index = data.copy(), index.copy()

        self.data, self.index = data, index

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        rows = []
        for idx, row in zip(self.index, self.data):
            rows.append('{}\t{}'.format(idx, row))
        return '\n'.join(rows)

    def __getitem__(self, item) -> IndexedDataset:
        if isinstance(item, (int, np.integer)):
            item = [item]

        return IndexedDataset(self.data[item], self.index[item], copy=False, check_data=False)

    def swap_rows(self, i: int, j: int) -> None:
        if i != j:
            self.index[[i, j]] = self.index[[j, i]]
            self.data[[i, j]] = self.data[[j, i]]

    def sample(self, size: Optional[int] = None) -> IndexedDataset:
        if size is None or size >= len(self):
            return self

        row_sample = np.random.choice(len(self), size, replace=False)
        return self[row_sample]
