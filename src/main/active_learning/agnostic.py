from .base import ActiveLearner


class RandomLearner(ActiveLearner):
    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def get_next(self, pool):
        return pool.sample_from_unlabeled()
