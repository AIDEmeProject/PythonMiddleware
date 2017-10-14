import numpy as np
from sklearn.utils import check_array
from sklearn.metrics.classification import precision_score, recall_score, accuracy_score, f1_score

from ..utils import label_all, check_points_and_labels
from ..datapool import DataPool
from ..metrics import MetricTracker
from ..version_space.base import VersionSpaceMixin


class ActiveLearner(object):
    """ 
        Class responsible for fitting a classifier and retrieving the next point from the data pool. 
    """
    def __init__(self):
        self.clf = None
        self.version_space = VersionSpaceMixin()

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
        scores.update(self.version_space.score())

        return scores

    def clear(self):
        if self.version_space is not None:
            self.version_space.clear()

    def initialize(self, data):
        pass

    def update(self, X, y):
        points, labels = check_points_and_labels(X, y)
        for point, label in zip(points, labels):
            self.version_space.update(point, label)

    def get_next(self, pool):
        raise NotImplementedError


def train(data, user, active_learner, initial_sampler):
    # check data
    data = check_array(np.atleast_2d(data), dtype=np.float64)

    # initialize
    user.clear()
    active_learner.clear()
    active_learner.initialize(data)
    points_init, labels_init = initial_sampler(data, user)

    # create data pool
    pool = DataPool(data)
    pool.update(points_init, labels_init)

    # train active_learner
    X_train, y_train = pool.get_labeled_data()
    active_learner.fit_classifier(X_train, y_train)
    active_learner.update(points_init.data, labels_init)

    # initialize tracker
    tracker = MetricTracker(skip=len(y_train))
    y_true = label_all(data, user)
    tracker.add_measurement(active_learner.score(data, y_true))

    while user.is_willing() and (not pool.has_labeled_all()):
        # get next point
        points = active_learner.get_next(pool)

        # label point
        labels = user.get_label(points)

        # update labeled/unlabeled sets
        pool.update(points, labels)

        # retrain active learner
        X_train, y_train = pool.get_labeled_data()
        active_learner.fit_classifier(X_train, y_train)
        active_learner.update(points.data, labels)

        # append new metrics
        tracker.add_measurement(active_learner.score(data, y_true))

    return tracker
