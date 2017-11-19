import numpy as np
from sklearn.metrics.classification import precision_score, recall_score, accuracy_score, f1_score
from ..version_space.base import VersionSpaceMixin


class ActiveLearner(object):
    """ 
        Class responsible for fitting a classifier and retrieving the next point from the data pool. 
    """
    def __init__(self, estimate_cut=False):
        self.clf = None
        self.version_space = VersionSpaceMixin()
        self.estimate_cut = estimate_cut

    def predict(self, X):
        return self.clf.predict(X)

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)

    def score(self, X, y_true):
        # classification scores
        y_pred = self.predict(X)
        scores = {
            'precision': precision_score(y_true, y_pred, labels=[-1,1]),
            'recall': recall_score(y_true, y_pred, labels=[-1, 1]),
            'accuracy': accuracy_score(y_true, y_pred),
            'fscore': f1_score(y_true, y_pred, labels=[-1, 1])
        }

        # version space scores
        if self.estimate_cut:
            scores.update(self.version_space.score())

        return scores

    def clear(self):
        self.version_space.clear()

    def initialize(self, data):
        pass

    def update(self, points, labels):
        for point, label in zip(points.values, labels):
            self.version_space.update(point, label)

    def get_next(self, pool):
        return pool.get_minimizer_over_unlabeled_data(self.ranker, size=1, sample_size=-1)

    def ranker(self, data):
        raise NotImplementedError
