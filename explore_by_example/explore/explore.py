from time import perf_counter

from sklearn.utils import check_X_y

from .partitioned import PartitionedDataset
from .user import User

from ..active_learning.dsm import DualSpaceModel


class PoolBasedExploration:
    def __init__(self, iters, initial_sampler, subsampling=float('inf'), callback=None, callback_skip=1):
        """
            :param iters: number of iterations to run
            :param initial_sampler: InitialSampler object
            :param callback: a callback function to be called at the end of every iteration. It must have the
            following signature:

                              callback(data, user, active_learner)

            The callback can optionally return a dictionary containing new metrics to be included.
            :param callback_skip: compute callback every callback_skip iterations only.
        """
        self.iters = iters
        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callback = callback
        self.callback_skip = callback_skip

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
        data = PartitionedDataset(X, copy=True)
        user = User(y, self.iters)
        active_learner.clear()

        iter, metrics = 0, []
        while self.__can_continue_exploration(data, user):
            metrics.append(self._run_single_iter(iter, data, user, active_learner))
            iter += 1

        return metrics

    def __can_continue_exploration(self, data, user):
        return user.is_willing and data.unknown_size > 0

    def _run_single_iter(self, iter, data, user, active_learner):
        """
            Run a single iteration of the active learning loop:
        """
        metrics = {}

        # find next point to label
        t0 = perf_counter()
        idx, X = self._get_next_point_to_label(data, user, active_learner)
        metrics['get_next_time'] = perf_counter() - t0

        # update labeled indexes
        labels = user.label(idx, X)
        data.move_to_labeled(idx, labels)

        metrics['labels'] = labels
        metrics['labeled_indexes'] = idx

        # fit active learning model
        t1 = perf_counter()
        active_learner.fit_data(data)
        metrics['fit_time'] = perf_counter() - t1

        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        if self.callback and (iter % self.callback_skip == 0 or not self.__can_continue_exploration(data, user)):
            callback_metrics = self.callback(iter, data, user, active_learner)

            if callback_metrics:
                metrics.update(callback_metrics)

        return metrics

    def _get_next_point_to_label(self, data, user, active_learner):
        """
           Get the next points to label. If not points have been labeled so far, the Initial Sampling procedure is run;
           otherwise, the most informative unlabeled data point is retrieved by the Active Learner

           :return: list containing the row numbers of the next point to label
       """

        if data.labeled_size == 0:
            idx_sample = self.initial_sampler(user.labels)
            return idx_sample, data.data[idx_sample]

        if data.unlabeled_size == 0:
            raise RuntimeError("The entire dataset has already been labeled!")

        return active_learner.next_points_to_label(data, self.subsampling)
