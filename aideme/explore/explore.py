#  Copyright (c) 2019 École Polytechnique
# 
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
# 
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
# 
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.

import math
from time import perf_counter

from sklearn.utils import check_X_y

from .partitioned import PartitionedDataset
from ..utils import assert_positive_integer


class PoolBasedExploration:
    def __init__(self, initial_sampler, subsampling=math.inf,
                 callback=None, callback_skip=1, print_callback_result=False,
                 convergence_criteria=None):
        """
            :param initial_sampler: InitialSampler object
            :param callback: a callback function to be called at the end of every iteration. It must have the
            following signature:

                              callback(data, user, active_learner)

            The callback can optionally return a dictionary containing new metrics to be included.
            :param callback_skip: compute callback every callback_skip iterations only.
        """
        assert_positive_integer(subsampling, 'subsampling', allow_inf=True)
        assert_positive_integer(callback_skip, 'callback_skip')

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

        data = PartitionedDataset(X, copy=True)
        iteration = Iteration(data, active_learner, self.subsampling)

        return [self._run(iteration, y) for _ in range(repeat)]

    def _run(self, iteration, y):
        iteration.clear()

        idx = self.initial_sampler(y)

        iter_metrics, metrics, converged = [], {}, False
        while not self.__converged(iteration, metrics):
            idx, metrics = iteration.advance(idx, y[idx])

            if self.__is_callback_computation_iter(iteration):
                metrics.update(self.__get_callback_metrics(iteration, y))

            iter_metrics.append(metrics)

        if not self.__is_callback_computation_iter(iteration):
            metrics.update(self.__get_callback_metrics(iteration, y))

        return iter_metrics

    def __is_callback_computation_iter(self, iteration):
        return (iteration.iter - 1) % self.callback_skip == 0

    def __get_callback_metrics(self, iteration, y):
        metrics = {}

        for callback in self.callbacks:
            callback_metrics = callback(iteration, y)

            if callback_metrics:
                metrics.update(callback_metrics)

        if self.print_callback_result:
            scores_str = ', '.join((k + ': ' + str(v) for k, v in metrics.items()))
            print('iter: {0}, {1}'.format(iteration.iter, scores_str))

        return metrics

    def __converged(self, iteration, metrics):
        return any((criterion(iteration, metrics) for criterion in self.convergence_criteria))


class Iteration:
    def __init__(self, data, active_learner, subsampling=math.inf):
        assert_positive_integer(subsampling, 'subsampling', allow_inf=True)

        self.data = data
        self.active_learner = active_learner
        self.subsampling = subsampling

        self.__iter = 0

    @staticmethod
    def __process_function(funcs):
        if not funcs:
            return []

        if callable(funcs):
            return [funcs]

        return funcs

    @property
    def iter(self):
        return self.__iter

    def clear(self):
        self.data.clear(copy=True)
        self.active_learner.clear()
        self.__iter = 0

    def advance(self, new_idx, new_labels):
        metrics = {}  # TODO: add 'iter': self.__iter ?

        self.data.move_to_labeled(new_idx, new_labels, 'user')
        metrics['labeled_indexes'] = new_idx
        metrics['labels'] = new_labels.tolist()

        # fit active learning model
        t0 = perf_counter()
        self._fit()
        metrics['fit_time'] = perf_counter() - t0

        # find next point to label
        t0 = perf_counter()
        idx, _ = self._get_next_point_to_label()
        metrics['get_next_time'] = perf_counter() - t0

        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        # update iteration counter
        self.__iter += 1

        return idx, metrics

    def predict_labels(self):
        return self.active_learner.predict(self.data.data)

    def _fit(self):
        self.active_learner.fit_data(self.data)

    def _get_next_point_to_label(self):
        if self.data.unlabeled_size == 0:
            raise RuntimeError("The entire dataset has already been labeled!")

        return self.active_learner.next_points_to_label(self.data, self.subsampling)
