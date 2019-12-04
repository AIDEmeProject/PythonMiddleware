import numpy as np


class PartitionedDataset:
    """
              0 <= i < infer_start    -> labeled partition
    infer_start <= i < unknown_start  -> infer partition
                   i >= unknown_start -> unknown partition
    """

    def __init__(self, data, copy=False):
        self.data = data

        self._data = data.copy() if copy else data

        self.__size = len(data)

        self._row_to_index = np.arange(self.__size)
        self._inferred_start = 0
        self._unknown_start = 0

        self._index_to_row = {i: i for i in range(self.__size)}

        self._labels = []
        self._label_tags = []

        self._previous_inferred_start = 0

    def __len__(self):
        return self.__size

    def __repr__(self):
        return "labeled: {0}, inferred: {1}, unknown: {2}".format(
            self._row_to_index[:self._inferred_start],
            self._row_to_index[self._inferred_start:self._unknown_start],
            self._row_to_index[self._unknown_start:]
        )

    def select_cols(self, data_cols, lb_cols):
        dataset = PartitionedDataset(self.data[:, data_cols], copy=False)
        dataset._data = self._data[:, data_cols]
        dataset._row_to_index = self._row_to_index
        dataset._inferred_start = self._inferred_start
        dataset._unknown_start = self._unknown_start
        dataset._index_to_row = self._index_to_row
        dataset._labels = [lb[lb_cols] for lb in self._labels]
        dataset._label_tags = self._label_tags
        dataset._previous_inferred_start = self._previous_inferred_start
        return dataset

    ##################
    # MOVING DATA
    ##################
    def move_to_labeled(self, indexes, labels, tag):
        self._previous_inferred_start = self._inferred_start

        for idx in indexes:
            self.__move_single(idx, True)

        self._labels.extend(labels)
        self._label_tags.extend([tag] * len(labels))

    def move_to_inferred(self, indexes):
        for idx in indexes:
            self.__move_single(idx, False)

    def __move_single(self, idx, to_labeled):
        pos = self._index_to_row[idx]

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
        idx_i, idx_j = self._row_to_index[i], self._row_to_index[j]
        self._row_to_index[i], self._row_to_index[j] = idx_j, idx_i
        self._index_to_row[idx_i], self._index_to_row[idx_j] = j, i
        self._data[[i, j]] = self._data[[j, i]]

    def remove_inferred(self):
        # flush inferred partition
        self._unknown_start = self._inferred_start

        # copy labeled indexes slice because we will modify list inplace
        labeled_idx = self._row_to_index[:self._inferred_start].copy()

        for idx, tag in zip(labeled_idx, self._label_tags):
            if tag != 'user':
                self.__move_right(idx)

        self._labels = [lb for lb, tag in zip(self._labels, self._label_tags) if tag == 'user']
        self._label_tags = ['user'] * len(self._labels)

    def __move_right(self, idx):
        pos = self._index_to_row[idx]

        if pos >= self._unknown_start:
            raise RuntimeError('Index {0} is already in unknown set.'.format(idx))

        if pos < self._inferred_start:
            self.__swap(pos, self._inferred_start - 1)
            pos = self._inferred_start - 1
            self._inferred_start -= 1
            self._previous_inferred_start -= 1

        if pos < self._unknown_start:
            self.__swap(pos, self._unknown_start - 1)
            self._unknown_start -= 1

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
        return self._row_to_index[:self._inferred_start], self._data[:self._inferred_start]

    @property
    def inferred(self):
        return self._row_to_index[self._inferred_start:self._unknown_start], self._data[self._inferred_start:self._unknown_start]

    @property
    def unknown(self):
        return self._row_to_index[self._unknown_start:], self._data[self._unknown_start:]

    @property
    def unlabeled(self):
        return self._row_to_index[self._inferred_start:], self._data[self._inferred_start:]

    ##################
    # SAMPLING
    ##################
    def sample_unknown(self, subsample=float('inf')):
        return self.__subsample(subsample, self._unknown_start, self.__size)

    def sample_inferred(self, subsample=float('inf')):
        return self.__subsample(subsample, self._inferred_start, self._unknown_start)

    def sample_unlabeled(self, subsample=float('inf')):
        return self.__subsample(subsample, self._inferred_start, self.__size)

    def __subsample(self, size, start, stop):
        remaining = stop - start

        if remaining == 0:
            raise RuntimeError("There are no points to sample from.")

        if size >= remaining:
            return self._row_to_index[start:stop], self._data[start:stop]

        row_sample = np.random.choice(np.arange(start, stop), size=size, replace=False)
        return self._row_to_index[row_sample], self._data[row_sample]

    ##################
    # LABELED DATA
    ##################
    @property
    def last_labeled_set(self):
        return self._data[self._previous_inferred_start:self._inferred_start], np.array(self._labels[self._previous_inferred_start:self._inferred_start])

    @property
    def training_set(self):
        return self._data[:self._inferred_start], np.array(self._labels)
