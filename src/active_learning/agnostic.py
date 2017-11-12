from .base import ActiveLearner
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
        return pool.sample_from_unlabeled()
