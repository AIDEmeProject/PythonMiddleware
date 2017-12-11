from random import randint
from numpy import argsort
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
        """
        Get all labeled points so far
        :return: pair (X,y) where X = labeled points and y = labels
        """
        return self.__data.loc[self.__labeled_rows], self.labels

    def get_positive_points(self):
        idx = [i for i, label in zip(self.__labeled_rows, self.__labels) if label == 1]
        return self.__data.loc[idx]

    def update(self, labels):
        """
        Update labeled/unlabeled sets.
        :param labels: pandas Dataframe/Series of new labels. Its index should match the labeled point's index.
        :return:
        """
        if len(labels.index.intersection(self.__labeled_rows)) > 0:
            raise RuntimeError("Trying to label the same point twice!")

        self.__labeled_rows.extend(labels.index)
        self.__labels.extend(labels)

    def sample_from_unlabeled(self):
        """
        Sample a single point uniformly from the unlabeled set through rejection sampling.
        """
        if self.has_labeled_all():
            raise RuntimeError("Cannot sample from empty set!")

        n = len(self.__data)
        while True:
            idx = self.__data.index[randint(0, n-1)]
            if idx not in self.__labeled_rows:
                return self.__data.loc[[idx]]

    def get_minimizer_over_unlabeled_data(self, ranker, sample_size=-1):
        """
        Computes the minimum of ranker function over the unlabeled set.

        :param ranker: function receiving a numpy matrix and returning the rank of each line
        :param sample_size: whether to restrict unlabeled set to a smaller sample
        """
        if self.has_labeled_all():
            raise RuntimeError("Empty unlabeled set!")

        # if unlabeled pool is too large, restrict search to sample
        data = self.__data if sample_size <= 0 else self.__data.sample(sample_size)

        thresholds = ranker(data.values)
        sorted_index = argsort(thresholds)

        for i in sorted_index:
            idx = data.index[i]
            if idx not in self.__labeled_rows:
                return self.__data.loc[[idx]]

