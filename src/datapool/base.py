from .unlabeled import UnlabeledSet
from .labeled import LabeledSet


class DataPool(object):
    def __init__(self, data):
        self.__labeled = LabeledSet()
        self.__unlabeled = UnlabeledSet(data)

    @property
    def size(self):
        return (len(self.__labeled), len(self.__unlabeled))

    @property
    def labeled_rows(self):
        return self.__unlabeled.labeled_rows

    def has_labeled_all(self):
        return len(self.__unlabeled) <= 0

    def get_labeled_set(self):
        return self.__labeled.to_array()

    def get_minimizer_over_unlabeled_data(self, ranker, size=1):
        return self.__unlabeled.get_minimizer(ranker, size)

    def update(self, points, labels):
        self.__labeled.append(points.values, labels)
        self.__unlabeled.update_labeled_rows(points.index)

    def sample_from_unlabeled(self):
        return self.__unlabeled.sample()

    def get_positive_points(self):
        return self.__labeled.get_positive_points()
