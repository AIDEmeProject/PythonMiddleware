import logging
import timeit

from src.config.utils import setup_logging
from src.datapool import DataPool
from src.utils import label_all

from src.main.metrics import MetricTracker

setup_logging("/Users/luciano/Projects/explore_by_example/src/config/logging.yml")

class Task:
    def __init__(self, data, user, learner):
        self.__data = data
        self.__user = user
        self.__learner = learner
        self.pool = DataPool(self.__data)

        # logging config
        self.logger = logging.getLogger("task")


    def clear(self):
        self.__user.clear()
        self.__learner.clear()
        self.pool.clear()

    def log_point(self, x, y, iter, name):
        log_string = "{0}\t{1}\t{2}\t{3}".format(name, iter, x, y)
        self.logger.info(log_string)

    def initialize(self):
        self.__learner.initialize(self.__data)

        # train active_learner
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X, y)

        # log data
        for x, y in zip(X.values, y.values):
            self.log_point(x, y, 0, 'IS')

    def get_score(self, y_true):
        return self.__learner.score(self.__data, y_true)

    def update_learner(self):
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X.iloc[[-1]], y.iloc[[-1]])

    def train(self, initial_sample):
        # clear any previous state
        self.clear()

        if initial_sample is not None:
            self.pool.update(initial_sample)
            self.initialize()

        else:
            self.__learner.initialize(self.__data)

            points = self.pool.sample_from_unlabeled()

            # label point
            labels = self.__user.get_label(points)

            # update labeled/unlabeled sets
            self.pool.update(labels)

            # retrain active learner
            self.update_learner()

        # initialize tracker
        tracker = MetricTracker(skip=self.pool.labeled_set_shape[0] - 1)
        y_true = label_all(self.__data, self.__user)
        tracker.add_measurement(self.get_score(y_true))

        i = 1
        while self.__user.is_willing() and (not self.pool.has_labeled_all()):
            # get next point
            t0 = timeit.default_timer()
            points = self.__learner.get_next(self.pool)
            get_next_time = timeit.default_timer() - t0

            # label point
            labels = self.__user.get_label(points)
            # update labeled/unlabeled sets
            t1 = timeit.default_timer()
            self.pool.update(labels)
            update_time = timeit.default_timer() - t1

            # retrain active learner
            t2 = timeit.default_timer()
            self.update_learner()
            retrain_time = timeit.default_timer() - t2

            iteration_time = timeit.default_timer() - t0
            # append new metrics

            for x, y in zip(points.values, labels.values):
                self.log_point(x, y, i, 'AL')

            scores = {
                'get_next_time': get_next_time,
                'update_time': update_time,
                'retrain_time': retrain_time,
                'iteration_time': iteration_time
            }
            scores.update(self.get_score(y_true))
            tracker.add_measurement(scores)
            i += 1

        return tracker


class Task2:
    def __init__(self, data, user, learner, initial_sampler):
        self.__data = data
        self.__user = user
        self.__learner = learner
        self.__initial_sampler = initial_sampler
        self.__pool = DataPool(self.__data)

        # logging config
        setup_logging("/Users/luciano/Projects/explore_by_example/src/config/logging.yml")
        self.logger = logging.getLogger("task")

    def train(self, initial_sample):
        # clear any previous state
        self.clear()

        # create data pool
        self.__learner.initialize(self.__data)

        # initial sampling
        initial_points = self.__data.loc[initial_sample.index]
        self.update_state(initial_points, initial_sample, 0, 'IS')

        i = 1
        while self.__user.is_willing() and (not self.__pool.has_labeled_all()):
            # get next point
            points = self.__learner.get_next(self.__pool)
            labels = self.__user.get_label(points)
            self.update_state(points, labels, i, 'AL')

            i += 1

    def clear(self):
        self.__pool.clear()
        self.__user.clear()
        self.__learner.clear()

    def update_state(self, points, labels, iteration, id):
        # update labeled/unlabeled sets
        self.__pool.update(labels)

        # update version space
        self.__learner.version_space.update(points, labels)

        # log labeled points
        for x, y in zip(points.values, labels.values):
            self.log_point(x, y, iteration, id)

    def log_point(self, x, y, iter, name):
        log_string = "{0}\t{1}\t{2}\t{3}".format(name, iter, x, y)
        self.logger.info(log_string)
