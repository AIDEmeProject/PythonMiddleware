import numpy as np


class PartitionedDataset:
    """
              0 <= i < infer_start    -> labeled partition
    infer_start <= i < unknown_start  -> infer partition
                   i >= unknown_start -> unknown partition
    """

    def __init__(self, data, copy=False):
        self.data = data.copy() if copy else data

        self.__size = len(data)

        self.__row_to_index = np.arange(self.__size)
        self._inferred_start = 0
        self._unknown_start = 0

        self.__index_to_row = {i: i for i in range(self.__size)}

        self.__labels = []

    def __len__(self):
        return self.__size

    @property
    def indexes(self):
        return self.__row_to_index

    ##################
    # MOVING DATA
    ##################
    def move_to_labeled(self, indexes, labels):
        for idx in indexes:
            self.__move_single(idx, True)

        self.__labels.extend(labels)

    def move_to_inferred(self, indexes):
        for idx in indexes:
            self.__move_single(idx, False)

    def __move_single(self, idx, to_labeled):
        if idx < 0 or idx >= self.__size:
            raise ValueError('Index {0} out of bounds for size {1}'.format(idx, self.__size))

        pos = self.__index_to_row[idx]

        if pos < self._inferred_start:
            raise RuntimeError('Index {0} is already in labeled set.'.format(idx))

        if not to_labeled and self._inferred_start <= pos < self._unknown_start:
            raise RuntimeError('Index {0} is already in inferred set.'.format(idx))

        if pos >= self._unknown_start:
            self.__swap(pos, self._unknown_start)
            pos = self._unknown_start
            self._unknown_start += 1

        if to_labeled and pos >= self._inferred_start:
            self.__swap(pos, self._inferred_start)
            self._inferred_start += 1

    def __swap(self, i, j):
        idx_i, idx_j = self.__row_to_index[i], self.__row_to_index[j]
        self.__row_to_index[i], self.__row_to_index[j] = idx_j, idx_i
        self.__index_to_row[idx_i], self.__index_to_row[idx_j] = j, i
        self.data[[i, j]] = self.data[[j, i]]

    ##################
    # SIZES
    ##################
    @property
    def labeled_size(self):
        return self._inferred_start

    @property
    def infer_size(self):
        return self._unknown_start - self._inferred_start

    @property
    def unknown_size(self):
        return self.__size - self._unknown_start

    @property
    def unlabeled_size(self):
        return self.__size - self._inferred_start

    ##################
    # DATA SLICING
    ##################
    @property
    def labeled(self):
        return self.__row_to_index[:self._inferred_start], self.data[:self._inferred_start]

    @property
    def inferred(self):
        return self.__row_to_index[self._inferred_start:self._unknown_start], self.data[self._inferred_start:self._unknown_start]

    @property
    def unknown(self):
        return self.__row_to_index[self._unknown_start:], self.data[self._unknown_start:]

    @property
    def unlabeled(self):
        return self.__row_to_index[self._inferred_start:], self.data[self._inferred_start:]

    ##################
    # SAMPLING
    ##################
    def sample_unknown(self, subsample=float('inf')):
        return self.__subsample(subsample, self._unknown_start)

    def sample_unlabeled(self, subsample=float('inf')):
        return self.__subsample(subsample, self._inferred_start)

    def __subsample(self, size, start):
        remaining = self.__size - start

        if remaining == 0:
            raise RuntimeError("There are no points to sample from.")

        if size >= remaining:
            return self.__row_to_index[start:], self.data[start:]

        row_sample = np.random.choice(np.arange(start, self.__size), size=size, replace=False)
        return self.__row_to_index[row_sample], self.data[row_sample]

    ##################
    # LABELED DATA
    ##################
    @property
    def labels(self):
        return np.asarray(self.__labels)

    @property
    def training_set(self):
        return self.data[:self._inferred_start], self.labels
