from random import randint
from .base import ActiveLearner
from ..datapool import Point
from ..version_space.svm import SVMVersionSpace


class RandomLearner(ActiveLearner):
    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def clear(self):
        pass

    def initialize(self, data):
        self.version_space = SVMVersionSpace(data.shape[1])

    def get_next(self, pool):
        n = len(pool)
        while True:
            idx = randint(0, n - 1)
            if idx not in pool.retrieved:
                return Point(index=[idx], data=pool[idx])
