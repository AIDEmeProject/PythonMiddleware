import math
from time import perf_counter

from sklearn.utils import check_X_y

from .partitioned import PartitionedDataset
from .user import User
from ..utils import assert_positive_integer


class PoolBasedExploration:
    def __init__(self, max_iter, initial_sampler, subsampling=math.inf,
                 callback=None, callback_skip=1, print_callback_result=False,
                 convergence_criteria=None):
        """
            :param max_iter: number of iterations to run
            :param initial_sampler: InitialSampler object
            :param callback: a callback function to be called at the end of every iteration. It must have the
            following signature:

                              callback(data, user, active_learner)

            The callback can optionally return a dictionary containing new metrics to be included.
            :param callback_skip: compute callback every callback_skip iterations only.
        """
        assert_positive_integer(max_iter, 'max_iter', allow_inf=True)
        assert_positive_integer(subsampling, 'subsampling', allow_inf=True)
        assert_positive_integer(callback_skip, 'callback_skip')

        self.max_iter = max_iter
        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = self.__process_function(callback)
        self.callback_skip = callback_skip
        self.print_callback_result = print_callback_result
        self.convergence_criteria = self.__process_function(convergence_criteria)

    @staticmethod
    def __process_function(callback):
        if not callback:
            return []

        if callable(callback):
            return [callback]

        return callback

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
        data, user = self.__initialize(X, y, active_learner)

        iter, iter_metrics, metrics = 0, [], {}
        while self.__will_continue_exploring(data, user, metrics):
            metrics = self._run_single_iter(iter, data, user, active_learner)
            iter_metrics.append(metrics)
            iter += 1

        return iter_metrics

    def __initialize(self, X, y, active_learner):
        data = PartitionedDataset(X, copy=True)
        user = User(y, self.max_iter)
        active_learner.clear()
        return data, user

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
        data.move_to_labeled(idx, labels, 'user')

        metrics['labels'] = labels.tolist()
        metrics['labeled_indexes'] = idx

        # fit active learning model
        t1 = perf_counter()
        active_learner.fit_data(data)
        metrics['fit_time'] = perf_counter() - t1

        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        # compute callback metrics
        if iter % self.callback_skip == 0 or not self.__will_continue_exploring(data, user, metrics):
            callback_metrics = self.__compute_callback_metrics(data, user, active_learner)
            self.__print_metrics(iter, callback_metrics)
            metrics.update(callback_metrics)

        return metrics

    def __will_continue_exploring(self, data, user, metrics):
        return user.is_willing and all((not criterion(data, metrics) for criterion in self.convergence_criteria))

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

    def __compute_callback_metrics(self, data, user, active_learner):
        metrics = {}

        for callback in self.callbacks:

            callback_metrics = callback(data.data, user.labels, active_learner)

            if callback_metrics:
                metrics.update(callback_metrics)

        return metrics

    def __print_metrics(self, iter, metrics):
        if self.print_callback_result:
            scores_str = ', '.join((k + ': ' + str(v) for k, v in metrics.items()))
            print('iter: {0}, {1}'.format(iter, scores_str))
