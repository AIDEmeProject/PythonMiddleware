from time import perf_counter

import numpy as np
from sklearn.utils import check_X_y


class PoolBasedExploration:
    def __init__(self, iters, initial_sampler, callback=None, subsampling=float('inf')):
        """
            :param iters: number of iterations to run
            :param initial_sampler: InitialSampler object
            :param callback: a callback function to be called at the end of every iteration. It must have the
            following signature:

                              callback(X, y, active_learner, labeled_indexes)

            The callback can optionally return a dictionary containing new metrics to be included.
        """
        self.iters = iters
        self.initial_sampler = initial_sampler
        self.callback = callback
        self.subsampling = subsampling

    def run(self, X, y, active_learner, repeat=1):
        """
            Run the Active Learning model over data, for a given number of iterations.

            :param X: data matrix
            :param y: labels array
            :param active_learner: Active Learning algorithm to run
            :param repeat: repeat exploration this number of times
            :return: a list of metrics collected after every iteration run. For each iteration we have a dictionary
            containing:

                    - Index of labeled points
                    - Timing measurements (fit, select next point, ...)
                    - Any metrics returned by the callback function
        """
        X, y = check_X_y(X, y, dtype="numeric", ensure_2d=True, multi_output=True, y_numeric=False,
                         copy=False, force_all_finite=True)

        return [self._run(X, y, active_learner) for _ in range(repeat)]

    def _run(self, X, y, active_learner):
        labeled_indexes = []
        return [self._run_single_iter(X, y, active_learner, labeled_indexes) for _ in range(self.iters)]

    def _run_single_iter(self, X, y, active_learner, labeled_indexes):
        """
            Run a single iteration of the active learning loop:
        """
        metrics = {}

        # find next point to label
        t0 = perf_counter()
        idx = self._get_next_point_to_label(X, y, active_learner, labeled_indexes)
        metrics['get_next_time'] = perf_counter() - t0

        # update labeled indexes
        labeled_indexes.extend(idx)
        metrics['labeled_indexes'] = idx

        # fit active learning model
        t1 = perf_counter()
        self._fit_model(X, y, active_learner, labeled_indexes)
        metrics['fit_time'] = perf_counter() - t1

        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        if self.callback:
            callback_metrics = self.callback(X, y, active_learner, labeled_indexes)

            if callback_metrics:
                metrics.update(callback_metrics)

        return metrics

    def _get_next_point_to_label(self, X, y, active_learner, labeled_indexes):
        """
           Get the next points to label. If not points have been labeled so far, the Initial Sampling procedure is run;
           otherwise, the most informative unlabeled data point is retrieved by the Active Learner

           :return: list containing the row numbers of the next point to label
       """

        # if there are no labeled_indexes, run initial sample
        if not labeled_indexes:
            return self.initial_sampler(y)

        # otherwise, select point by running the active learning procedure
        row_sample, X_sample = self.__subsample_data(X)

        for _, row_number in sorted(zip(active_learner.rank(X_sample), row_sample)):
            if row_number not in labeled_indexes:
                return [row_number]

        raise RuntimeError("The entire dataset has already been labeled!")

    def _fit_model(self, X, y, active_learner, labeled_indexes):
        """
            Fit the active learning model over the labeled data
        """
        X_train, y_train = X[labeled_indexes], y[labeled_indexes]
        active_learner.fit(X_train, y_train)

    def __subsample_data(self, X):
        idx = np.arange(len(X))

        if self.subsampling >= len(X):
            return idx, X

        idx = np.random.choice(idx, size=self.subsampling, replace=False)
        return idx, X[idx]
