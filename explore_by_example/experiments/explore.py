from time import perf_counter

import sklearn
from numpy import argsort


class PoolBasedExploration:
    def __init__(self, iter, initial_sampler, metric=None, plot=None):
        """
        :param iter: number of iterations of the active learning algorithm to run
        :param initial_sampler: InitialSampler object
        :param metric: accuracy metric to be computed every iteration
        :param plot: plotting function for each iteration
        """
        self.iter = iter
        self.initial_sampler = initial_sampler
        self.metric = metric
        self.plot = plot

    def _find_minimum_over_unlabeled_data(self, X, learner, labeled_indexes=None):
        """
        Get next point to label. We retrieve the "lowest ranked unlabeled point" in the dataset X.

        :param X: data matrix
        :param labeled_indexes: rows to ignore when retrieving the minimum.
        :return: row number of next point to be labeled
        """
        if labeled_indexes is None:
            labeled_indexes = []

        for row_number in argsort(learner.rank(X)):
            if row_number not in labeled_indexes:
                return row_number

        raise RuntimeError("The entire dataset has already been labeled!")


    def _run_iter(self, iter, X, y, learner, labeled_indexes, metrics):
        t0 = perf_counter()
        learner.fit(X[labeled_indexes], y[labeled_indexes])
        metrics[iter]['fit_time'] = perf_counter() - t0

        if self.metric:
            t0 = perf_counter()
            y_pred = learner.predict(X)
            metrics[iter]['predict_time'] = perf_counter() - t0
            metrics[iter]['accuracy'] = self.metric(y_true=y, y_pred=y_pred)

        if self.plot:
            self.plot(X, y, learner, labeled_indexes)

    def run(self, X, y, learner):
        """
        Run Active Learning model over data, for a given number of iterations.

        :param X: data matrix
        :param y: labels array
        :param learner: Active Learning algorithm to run
        :return: labeled points chosen by the algorithm
        """
        X, y = sklearn.utils.check_X_y(X, y)

        metrics = [dict() for _ in range(self.iter+1)]

        # fit model over initial sample
        t0 = perf_counter()
        labeled_indexes = self.initial_sampler(y)
        metrics[0]['index'] = labeled_indexes.copy()

        self._run_iter(0, X, y, learner, labeled_indexes, metrics)
        metrics[0]['iter_time'] = perf_counter() - t0

        # run iterations
        for i in range(self.iter):
            # get next point to label
            t0 = perf_counter()
            idx = self._find_minimum_over_unlabeled_data(X, learner, labeled_indexes)
            metrics[i+1]['get_next_time'] = perf_counter() - t0
            metrics[i+1]['index'] = idx
            labeled_indexes.append(idx)

            self._run_iter(i+1, X, y, learner, labeled_indexes, metrics)
            metrics[i+1]['iter_time'] = perf_counter() - t0

        return metrics