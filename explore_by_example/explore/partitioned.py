import numpy as np


class PartitionedDataset:
    def __init__(self, data):
        self.data = data
        self.__labels = []
        self.indexes = PartitionedIndex(len(data))

    @property
    def labeled_size(self):
        return self.indexes.num_labeled

    @property
    def unlabeled_size(self):
        return self.indexes.num_unlabeled

    @property
    def labels(self):
        return np.asarray(self.__labels)

    @property
    def labeled(self):
        return self.data[self.indexes.labeled], self.labels

    def unlabeled(self, size=float('inf')):
        idx_sample = self.indexes.unlabeled(size)
        return idx_sample, self.data[idx_sample]

    def add_labeled_indexes(self, idx, label):
        self.indexes.add_labeled_indexes(idx)
        self.__labels.extend(label)


class PartitionedIndex:
    def __init__(self, size):
        self.__size = size
        self.__labeled_indexes = []
        self.__unlabeled_rows_mask = np.full(shape=(self.__size,), fill_value=True)

    def __repr__(self):
        return "labeled: {0} | unlabeled: {1}".format(self.labeled, self.unlabeled)

    def __len__(self):
        return self.__size

    def add_labeled_indexes(self, indices):
        for idx in indices:
            self.__add_single(idx)

    def __add_single(self, idx):
        if idx < 0 or idx >= self.__size:
            raise ValueError('Index {0} out of bounds for size {1}'.format(idx, self.__size))

        if not self.__unlabeled_rows_mask[idx]:
            raise RuntimeError('Index {0} is already in labeled set.'.format(idx))

        self.__labeled_indexes.append(idx)
        self.__unlabeled_rows_mask[idx] = False

    @property
    def num_labeled(self):
        return len(self.__labeled_indexes)

    @property
    def num_unlabeled(self):
        return self.__size - self.num_labeled

    @property
    def labeled(self):
        return self.__labeled_indexes

    def unlabeled(self, size=float('inf')):
        unlabeled_idx = np.arange(self.__size)[self.__unlabeled_rows_mask]

        if size >= self.num_unlabeled:
            return unlabeled_idx

        return np.random.choice(unlabeled_idx, size=size, replace=False)
