from time import perf_counter

from sklearn.utils import check_X_y

from .partitioned import PartitionedDataset
from .user import User


class PoolBasedExploration:
    def __init__(self, iters, initial_sampler, callback=None, subsampling=float('inf')):
        """
            :param iters: number of iterations to run
            :param initial_sampler: InitialSampler object
            :param callback: a callback function to be called at the end of every iteration. It must have the
            following signature:

                              callback(data, user, active_learner)

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
        data = PartitionedDataset(X)
        user = User(y, self.iters)

        active_learner.clear()

        metrics = []
        while user.is_willing:
            metrics.append(self._run_single_iter(data, user, active_learner))

        return metrics

    def _run_single_iter(self, data, user, active_learner):
        """
            Run a single iteration of the active learning loop:
        """
        metrics = {}

        # find next point to label
        t0 = perf_counter()
        idx = self._get_next_point_to_label(data, user, active_learner)
        metrics['get_next_time'] = perf_counter() - t0

        # update labeled indexes
        labels = user.label(idx)
        data.add_labeled_indexes(idx, labels)
        metrics['labels'] = labels
        metrics['labeled_indexes'] = idx

        # fit active learning model
        t1 = perf_counter()
        self._fit_model(data, active_learner)
        metrics['fit_time'] = perf_counter() - t1

        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        if self.callback:
            callback_metrics = self.callback(data, user, active_learner)

            if callback_metrics:
                metrics.update(callback_metrics)

        return metrics

    def _get_next_point_to_label(self, data, user, active_learner):
        """
           Get the next points to label. If not points have been labeled so far, the Initial Sampling procedure is run;
           otherwise, the most informative unlabeled data point is retrieved by the Active Learner

           :return: list containing the row numbers of the next point to label
       """

        if not data.num_labeled > 0:
            return self.initial_sampler(user.labels)

        if not data.num_unlabeled > 0:
            raise RuntimeError("The entire dataset has already been labeled!")

        # otherwise, select point by running the active learning procedure
        idx_sample, X_sample = data.unlabeled(self.subsampling)

        idx_to_label = active_learner.next_points_to_label(X_sample)

        return idx_sample[idx_to_label]


    def _fit_model(self, data, active_learner):
        """
            Fit the active learning model over the labeled data
        """
        X_train, y_train = data.labeled
        active_learner.fit(X_train, y_train)
