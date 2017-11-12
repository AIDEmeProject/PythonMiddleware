from numpy import argsort
from random import randint


class UnlabeledSet:
    """
        In-memory or In-disk config for storing the remaining unlabeled points. It provides data access methods and
        utility functions.
    """
    def __init__(self, data):
        self.__data = data
        self.__labeled_rows = set()

    def __len__(self):
        return len(self.__data) - len(self.__labeled_rows)

    @property
    def labeled_rows(self):
        return self.__labeled_rows

    def is_empty(self):
        return len(self) == 0

    def update_labeled_rows(self, index):
        if hasattr(index, '__iter__'):
            self.__labeled_rows.update(index)
        else:
            self.__labeled_rows.add(index)

    def sample(self):
        if self.is_empty():
            raise RuntimeError("Cannot sample from empty set!")

        n = len(self) - 1
        while True:
            idx = self.__data.index[randint(0, n)]
            if idx not in self.__labeled_rows:
                return self.__data.loc[[idx]]

    def get_minimizer(self, ranker, size=1):
        """
            Retrieves 'size' unlabeled point minimizing the ranker function
            :param: ranker:
        """
        if not (isinstance(size, int) and size > 0):
            raise ValueError("Size must be a positive integer!")

        to_retrieve = []
        thresholds = ranker(self.__data.values)
        sorted_index = argsort(thresholds)

        for idx in sorted_index:
            if idx not in self.__labeled_rows:
                to_retrieve.append(idx)

            if len(to_retrieve) == size:
                return self.__data.loc[to_retrieve]
