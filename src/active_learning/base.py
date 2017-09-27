import numpy as np
from sklearn.utils import check_array
from ..utils import label_all
from ..datapool import DataPool
from ..metrics import MetricTracker


class ActiveLearner(object):
    """ 
        Class responsible for fitting a classifier and retrieving the next point from the data pool. 
    """
    def __init__(self):
        self.clf = None

    def clear(self):
        pass

    def predict(self, X):
        return self.clf.predict(X)

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)

    def initialize(self, data):
        pass

    def update(self, X, y):
        pass

    def get_next(self, pool):
        raise NotImplementedError


def train(data, user, active_learner, initial_sampler):
    # check data
    data = check_array(np.atleast_2d(data), dtype=np.float64)

    # initialize
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
    tracker = MetricTracker(metrics_list=['f1', 'accuracy'], skip=len(y_train))
    tracker.add_measurement(y_true=label_all(data, user), y_pred=active_learner.predict(data))

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
        tracker.add_measurement(y_true=label_all(data, user), y_pred=active_learner.predict(data))

    return tracker
