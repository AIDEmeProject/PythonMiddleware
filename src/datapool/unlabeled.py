from collections import namedtuple
import numpy as np


Point = namedtuple('Point', ['index', 'data'])


class UnlabeledSet(object):
    """
        In-memory or In-disk dataset for storing the remaining unlabeled points. It provides data access methods and
        utility functions.
    """
    def __init__(self, data):
        self.__data = data
        self.__retrieved = []

    def __getitem__(self, item):
        return self.__data[item]

    @property
    def retrieved(self):
        return self.__retrieved

    def has_removed_all(self):
        return len(self.__retrieved) == len(self.__data)

    def remove(self, index):
        self.__retrieved.extend(index)

    def find_minimizer(self, ranker):
        """ 
            Retrieves an unlabeled point that minimizes the ranker function 
            :param: ranker: 
        """
        sorted_idx = np.argsort(ranker(self.__data))
        for idx in sorted_idx:
            if idx not in self.__retrieved:
                return Point(index=[idx], data=self.__data[idx, :])
