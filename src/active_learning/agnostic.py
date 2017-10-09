from random import randint
from .base import ActiveLearner
from ..datapool import Point


class RandomLearner(ActiveLearner):
    def __init__(self, clf, vs=None):
        super().__init__()
        self.clf = clf
        if vs is not None:
            self.version_space = vs

    def get_next(self, pool):
        n = len(pool)
        while True:
            idx = randint(0, n - 1)
            if idx not in pool.retrieved:
                return Point(index=[idx], data=pool[idx])
