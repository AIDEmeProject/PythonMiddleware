from time import time

from src.datapool import DataPool
from src.initial_sampling import StratifiedSampler
from src.metrics import MetricTracker, MetricStorage
from src.utils import label_all


class Task:
    def __init__(self, data, user, learner):
        self.__data = data
        self.__user = user
        self.__learner = learner
        self.__initial_sampler = StratifiedSampler(pos=1, neg=1)

        self.pool = DataPool(self.__data)

    def clear(self):
        self.__user.clear()
        self.__learner.clear()
        self.pool.clear()

    def initialize(self):
        # create data pool
        self.__learner.initialize(self.__data)

        # initialize
        sample = self.__initial_sampler(self.__data, self.__user)
        self.pool.update(sample)

        # train active_learner
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X, y)

    def get_score(self, y_true):
        return self.__learner.score(self.__data, y_true)

    def update_learner(self):
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X.iloc[[-1]], y.iloc[[-1]])

    def main_loop(self):
        # get next point
        t0 = time()
        points = self.__learner.get_next(self.pool)
        get_next_time = time() - t0

        # label point
        labels = self.__user.get_label(points)
        # update labeled/unlabeled sets
        t1 = time()
        self.pool.update(labels)
        update_time = time() - t1

        # retrain active learner
        t2 = time()
        self.update_learner()
        retrain_time = time() - t2

        iteration_time = time() - t0
        # append new metrics
        return {
            'get_next_time': get_next_time,
            'update_time': update_time,
            'retrain_time': retrain_time,
            'iteration_time': iteration_time
        }

    def train(self):
        # clear any previous state
        self.clear()
        self.initialize()

        # initialize tracker
        tracker = MetricTracker(skip=self.pool.labeled_set_shape[0] - 1)
        y_true = label_all(self.__data, self.__user)
        tracker.add_measurement(self.get_score(y_true))

        while self.__user.is_willing() and (not self.pool.has_labeled_all()):
            scores = self.main_loop()
            scores.update(self.get_score(y_true))
            tracker.add_measurement(scores)

        return tracker

    def get_average_performance(self, repeat):
        storage = MetricStorage()

        # train learner for several iterations
        for _ in range(repeat):
            storage.persist(self.train())

        # compute average performance
        return storage.average_performance()
