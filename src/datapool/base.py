from .unlabeled import UnlabeledSet
from .labeled import LabeledSet


class DataPool(object):
    def __init__(self, data):
        self.__labeled = LabeledSet()
        self.__unlabeled = UnlabeledSet(data)
        self.__size = len(data)

    def __len__(self):
        return self.__size

    def __getitem__(self, item):
        return self.__unlabeled[item]

    @property
    def labeled_size(self):
        return len(self.__labeled)

    @property
    def retrieved(self):
        return self.__unlabeled.retrieved

    def has_labeled_all(self):
        return self.__unlabeled.has_removed_all()

    def get_labeled_data(self):
        return self.__labeled.to_array()

    def find_minimizer(self, ranker, threshold=None):
        return self.__unlabeled.find_minimizer(ranker, threshold)

    def update(self, points, labels):
        self.__labeled.append(points.data, labels)
        self.__unlabeled.remove(points.index)
