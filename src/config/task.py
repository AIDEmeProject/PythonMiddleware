from src.datapool import DataPool
from src.metrics import MetricTracker, MetricStorage
from src.utils import label_all
from time import time


class Task:
    def __init__(self, data, user, learner, initial_sampler, repeat):
        self.__data = data
        self.__user = user
        self.__learner = learner
        self.__initial_sampler = initial_sampler
        self.__repeat = repeat

        self.pool = None

    def clear(self):
        self.__user.clear()
        self.__learner.clear()
        self.pool = DataPool(self.__data)

    def initialize(self):
        # create data pool
        self.__learner.initialize(self.__data)

        # initialize
        points_init, labels_init = self.__initial_sampler(self.__data, self.__user)
        self.pool.update(points_init, labels_init)

        # train active_learner
        self.__learner.fit_classifier(points_init.values, labels_init)
        self.__learner.update(points_init, labels_init)

    def get_score(self, y_true):
        return self.__learner.score(self.__data, y_true)

    def update_learner(self):
        X_train, y_train = self.pool.get_labeled_set()
        #print('update: ', X_train, y_train)
        self.__learner.fit_classifier(X_train, y_train)
        self.__learner.update(X_train.iloc[[-1]], y_train[-1])


    def main_loop(self):
        # get next point
        t0 = time()
        points = self.__learner.get_next(self.pool)
        get_next_time = time() - t0

        # label point
        labels = self.__user.get_label(points)
        # update labeled/unlabeled sets
        t1 = time()
        self.pool.update(points, labels)
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
        tracker = MetricTracker(skip=self.pool.size[0] - 1)
        y_true = label_all(self.__data, self.__user)
        tracker.add_measurement(self.get_score(y_true))

        while self.__user.is_willing() and (not self.pool.has_labeled_all()):
            scores = self.main_loop()
            scores.update(self.get_score(y_true))
            tracker.add_measurement(scores)

        return tracker

    def get_average_performance(self):
        storage = MetricStorage()

        # train learner for several iterations
        for _ in range(self.__repeat):
            storage.persist(self.train())

        # compute average performance
        return storage.average_performance()


if __name__ == '__main__':
    from src.config import get_dataset_and_user
    from src.active_learning.svm import SimpleMargin, SolverMethod, OptimalMargin
    from src.active_learning.agnostic import RandomLearner
    from src.initial_sampling import FixedSizeStratifiedSampler
    from src.showdown import Showdown
    from sklearn.svm import SVC

    X_housing, user_housing = get_dataset_and_user('housing')
    user_housing.max_iter = 10

    active_learners_list = [
        #("random", RandomLearner(SVC(C=1000, kernel='linear'))),
        #("linearSVM", SimpleMargin(C=1000, kernel='linear', fit_intercept=False)),
        ('optimalMargin', OptimalMargin(C=1000, fit_intercept=False))
    ]

    datasets_list = [
        ("housing", X_housing, user_housing)
    ]

    times = 1
    initial_sampler = FixedSizeStratifiedSampler(sample_size=2)
    showdown = Showdown(times, initial_sampler)
    output = showdown.run(datasets_list, active_learners_list)