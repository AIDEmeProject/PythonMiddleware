import timeit

from src.main.datapool import DataPool
from src.main.utils import label_all

from src.main.metrics import MetricTracker

class Task:
    def __init__(self, data, user, learner):
        self.__data = data
        self.__user = user
        self.__learner = learner
        self.pool = DataPool(self.__data)


    def clear(self):
        self.__user.clear()
        self.__learner.clear()
        self.pool.clear()

    def initialize(self):
        self.__learner.initialize(self.__data)

        # train active_learner
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X, y)

    def get_score(self, y_true):
        scores = self.__learner.score(self.__data, y_true)

        X,y = self.pool.get_labeled_set()
        scores_labeled = self.__learner.score(X, y)

        scores['Labeled Set F-Score'] = scores_labeled['F-Score']
        scores['Imbalance'] = 100.0*((y == 1).sum())/len(y)

        return scores

    def update_learner(self):
        X, y = self.pool.get_labeled_set()
        self.__learner.fit_classifier(X, y)
        self.__learner.update(X.iloc[[-1]], y.iloc[[-1]])

    def train(self, initial_sample):
        # clear any previous state
        self.clear()

        self.pool.update(initial_sample)
        self.initialize()

        # initialize tracker
        tracker = MetricTracker()
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

            scores = {
                'Get Next Time': get_next_time,
                'Update Time': update_time,
                'Retrain Time': retrain_time,
                'Iteration Time': iteration_time
            }
            scores.update(self.get_score(y_true))
            tracker.add_measurement(scores)
            i += 1

        return tracker.to_dataframe()