from random import randint
from numpy import argsort, array
from pandas import Series

class DataPool:
    def __init__(self, data):
        self.__data = data
        self.__labeled_rows = []
        self.__labels = []

    @property
    def labeled_rows(self):
        return self.__labeled_rows

    @property
    def labels(self):
        return Series(self.__labels, index=self.__labeled_rows)

    @property
    def labeled_set_shape(self):
        return (len(self.__labeled_rows), self.__data.shape[1])

    @property
    def unlabeled_set_shape(self):
        return (len(self.__data) - len(self.__labeled_rows), self.__data.shape[1])

    def clear(self):
        self.__labeled_rows = []
        self.__labels = []

    def has_labeled_all(self):
        return self.unlabeled_set_shape[0] <= 0

    def get_labeled_set(self):
        return self.__data.loc[self.__labeled_rows], self.labels

    def get_positive_points(self):
        idx = [i for i, label in zip(self.__labeled_rows, self.__labels) if label == 1]
        return self.__data.loc[idx]

    def update(self, labels):
        if len(labels.index.intersection(self.__labeled_rows)) > 0:
            raise RuntimeError("Trying to label the same point twice!")

        self.__labeled_rows.extend(labels.index)
        self.__labels.extend(labels)

    def sample_from_unlabeled(self):
        if self.has_labeled_all():
            raise RuntimeError("Cannot sample from empty set!")

        n = len(self.__data)
        while True:
            idx = self.__data.index[randint(0, n-1)]
            if idx not in self.__labeled_rows:
                return self.__data.loc[[idx]]

    def get_minimizer_over_unlabeled_data(self, ranker, size=1, sample_size=-1):
        if not (isinstance(size, int) and size > 0):
            raise ValueError("Size must be a positive integer!")

        if len(self.__data) - len(self.__labeled_rows) < size:
            raise ValueError("Size larger than unlabeled set size!")

        to_retrieve = []
        data = self.__data if sample_size <= 0 else self.__data.sample(sample_size)
        thresholds = ranker(data.values)
        sorted_index = argsort(thresholds)

        for i in sorted_index:
            idx = data.index[i]
            if idx not in self.__labeled_rows:
                to_retrieve.append(idx)

            if len(to_retrieve) == size:
                return self.__data.loc[to_retrieve]

