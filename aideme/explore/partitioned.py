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

from typing import TYPE_CHECKING, Iterable, Tuple, List, Optional

import numpy as np
import sklearn.utils

from . import LabeledSet

if TYPE_CHECKING:
    pass


class PartitionedDataset:  # TODO: how can we add partition information? (factorization)
    def __init__(self, dataset: IndexedDataset, copy: bool = False):
        self.data = dataset
        self.__dataset = dataset.copy() if copy else dataset
        self.__index_to_row = {idx: i for i, idx in enumerate(self.__dataset.index)}
        self.__labeled_set = LabeledSet.empty()
        self.__inferred_start = 0
        self.__unknown_start = 0
        self.__non_user_labeled_indexes: List = []
        self.__last_user_labeled_set = LabeledSet.empty()

    def __len__(self):
        return len(self.__dataset)

    def __repr__(self):
        return self.__dataset.__repr__()

    def clear(self) -> None:
        self.__dataset = self.data.copy()
        self.__index_to_row = {idx: i for i, idx in enumerate(self.__dataset.index)}
        self.__labeled_set = LabeledSet.empty()
        self.__inferred_start = 0
        self.__unknown_start = 0
        self.__non_user_labeled_indexes = []
        self.__last_user_labeled_set = LabeledSet.empty()

    ##################
    # MOVING
    ##################
    def move_to_labeled(self, labeled_set: LabeledSet, user_labeled: bool = True) -> None:
        for idx in labeled_set.index:
            pos = self.__index_to_row[idx]

            if pos >= self.__inferred_start:
                self.__move_left(pos, to_labeled=True)
            else:
                self.__raise_error(idx, pos, 'labeled')

        self.__labeled_set = self.__labeled_set.concat(labeled_set)

        if user_labeled:
            self.__last_user_labeled_set = labeled_set
        else:
            self.__non_user_labeled_indexes.extend(labeled_set.index)

    def move_to_inferred(self, indexes: Iterable) -> None:
        for idx in indexes:
            pos = self.__index_to_row[idx]

            if pos >= self.__unknown_start:
                self.__move_left(pos, to_labeled=False)
            elif pos < self.__inferred_start:
                self.__move_right(pos, to_unknown=False)
            else:
                self.__raise_error(idx, pos, 'inferred')

    def move_to_unknown(self, indexes: Iterable) -> None:
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

    def __swap_rows(self, i, j):
        idx_i, idx_j = self.__dataset.index[i], self.__dataset.index[j]
        self.__dataset.swap_rows(i, j)
        self.__index_to_row[idx_i], self.__index_to_row[idx_j] = j, i

    def __raise_error(self, idx, pos: int, partition_name: str) -> None:
        raise ValueError("Index {}, at position {}, is already in {} set.".format(idx, pos, partition_name))

    def remove_inferred(self) -> None:
        self.__unknown_start = self.__inferred_start  # flush inferred partition

        self.move_to_unknown(self.__non_user_labeled_indexes)

        rows_to_keep = np.setdiff1d(self.__labeled_set.index, self.__non_user_labeled_indexes, assume_unique=True)
        self.__labeled_set = self.__labeled_set[rows_to_keep]

        self.__non_user_labeled_indexes = []  # only user-labeled points remain in labeled partition

    ##################
    # SIZES
    ##################
    @property
    def labeled_size(self) -> int:
        return self.__inferred_start

    @property
    def infer_size(self) -> int:
        return self.__unknown_start - self.__inferred_start

    @property
    def unknown_size(self) -> int:
        return len(self) - self.__unknown_start

    @property
    def unlabeled_size(self) -> int:
        return len(self) - self.__inferred_start

    ##################
    # DATA SLICING
    ##################
    @property
    def labeled(self) -> IndexedDataset:
        return self.__dataset[:self.__inferred_start]

    @property
    def inferred(self) -> IndexedDataset:
        return self.__dataset[self.__inferred_start:self.__unknown_start]

    @property
    def unknown(self) -> IndexedDataset:
        return self.__dataset[self.__unknown_start:]

    @property
    def unlabeled(self) -> IndexedDataset:
        return self.__dataset[self.__inferred_start:]

    ##################
    # SAMPLING
    ##################
    # TODO: refactor the functions below?
    def sample_unknown(self, subsample: Optional[int] = None) -> IndexedDataset:
        return self.__subsample(subsample, self.__unknown_start)

    def sample_unlabeled(self, subsample: Optional[int] = None) -> IndexedDataset:
        return self.__subsample(subsample, self.__inferred_start)

    def __subsample(self, size: Optional[int], start: int) -> IndexedDataset:
        remaining = len(self) - start

        if size is None:
            size = remaining

        if remaining == 0:
            raise RuntimeError("There are no points to sample from.")

        if size >= remaining:
            return self.__dataset[start:]

        row_sample = start + np.random.choice(remaining, size=size, replace=False)
        return self.__dataset[row_sample]

    ##################
    # LABELED DATA
    ##################
    def last_training_set(self, get_partial=False) -> Tuple[np.ndarray, np.ndarray]:
        rows = [self.__index_to_row[idx] for idx in self.__last_user_labeled_set.index]
        return self.__dataset.data[rows], self.__last_user_labeled_set.partial if get_partial else self.__last_user_labeled_set.labels

    def training_set(self, get_partial=False) -> Tuple[np.ndarray, np.ndarray]:
        return self.labeled.data, self.__labeled_set.partial if get_partial else self.__labeled_set.labels

    @property
    def labeled_set(self) -> LabeledSet:
        return self.__labeled_set

    def select_cols(self, data_cols, lb_cols):
        #TODO: implement this
        pass


class IndexedDataset:
    def __init__(self, X, index=None):
        self.data: np.ndarray
        self.index: np.ndarray

        if index is None:
            self.data = sklearn.utils.check_array(X)
            self.index = np.arange(len(X))
        else:
            self.data, self.index = sklearn.utils.check_X_y(X, index)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item) -> IndexedDataset:
        if isinstance(item, (int, np.integer)):
            item = [item]

        return IndexedDataset(self.data[item], self.index[item])

    @property
    def dim(self) -> int:
        return self.data.shape[1]

    def copy(self) -> IndexedDataset:
        return IndexedDataset(self.data.copy(), self.index.copy())

    def swap_rows(self, i: int, j: int) -> None:
        if i != j:
            self.index[[i, j]] = self.index[[j, i]]
            self.data[[i, j]] = self.data[[j, i]]

    def sample(self, size: Optional[int] = None) -> IndexedDataset:
        if size is None or size >= len(self):
            return self

        row_sample = np.random.choice(len(self), size, replace=False)
        return self[row_sample]