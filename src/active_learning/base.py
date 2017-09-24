import numpy as np
from sklearn.utils import check_array
from ..datapool import DataPool


class ActiveLearner(object):
    """ 
        Class responsible for fitting a classifier and retrieving the next point from the data pool. 
    """
    def __init__(self):
        self.clf = None

    def predict(self, X):
        return self.clf.predict(X)

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)

    def initialize(self, data):
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

    # main loop
    while user.is_willing() and (not pool.has_labeled_all()):
        points = active_learner.get_next(pool)
        labels = user.get_label(points)
        pool.update(points, labels)

        X_train, y_train = pool.get_labeled_data()
        active_learner.fit_classifier(X_train, y_train)
