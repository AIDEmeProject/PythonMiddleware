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

from .partitioned import PartitionedDataset
from ..utils import assert_positive_integer


class PoolBasedExploration:
    def __init__(self, initial_sampler=None, subsampling=math.inf,
                 callback=None, callback_skip=1, print_callback_result=False,
                 convergence_criteria=None):
        """
            :param initial_sampler: InitialSampler object. If None, no initial sampling will be done.
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

    def run(self, X, user, active_learner, repeat=1):
        """
            Run the Active Learning model over data, for a given number of iterations.

            :param X: data matrix
            :param user: user instance
            :param active_learner: Active Learning algorithm to run
            :param repeat: repeat exploration this number of times
            :return: a list of metrics collected after every iteration run. For each iteration we have a dictionary
            containing:

                    - Index of labeled points
                    - Timing measurements (fit, select next point, ...)
                    - Any metrics returned by the callback function
        """
        data = PartitionedDataset(X, copy=True)
        iteration = Iteration(data, active_learner, self.initial_sampler, self.subsampling)

        return [self._run(iteration, user) for _ in range(repeat)]

    def _run(self, iteration, user):
        iteration.clear()
        user.clear()

        iter_metrics, metrics = [], {}
        idx = iteration.initial_sampling_advance(user, metrics)

        while user.is_willing and not self.__converged(iteration, metrics):
            partial_labels, final_labels = user.label(idx, iteration.data.data[idx])
            idx, metrics = iteration.advance(idx, partial_labels, final_labels, user)

            if iteration.is_exploration_phase and self.__is_callback_computation_iter(iteration):
                metrics.update(self.__get_callback_metrics(iteration, user))

            iter_metrics.append(metrics)

        if iteration.is_exploration_phase and not self.__is_callback_computation_iter(iteration):
            metrics.update(self.__get_callback_metrics(iteration, user))

        return iter_metrics

    def __is_callback_computation_iter(self, iteration):
        return iteration.iter % self.callback_skip == 0

    def __get_callback_metrics(self, iteration, user):
        metrics = {}

        for callback in self.callbacks:
            callback_metrics = callback(iteration, user)

            if callback_metrics:
                metrics.update(callback_metrics)

        if self.print_callback_result:
            scores_str = ', '.join((k + ': ' + str(v) for k, v in metrics.items()))
            print('iter: {0}, {1}'.format(iteration.iter, scores_str))

        return metrics

    def __converged(self, iteration, metrics):
        return any((criterion(iteration, metrics) for criterion in self.convergence_criteria))


class Iteration:
    def __init__(self, data, active_learner, initial_sampler, subsampling=math.inf):
        assert_positive_integer(subsampling, 'subsampling', allow_inf=True)

        self.data = data
        self.active_learner = active_learner
        self.initial_sampler = initial_sampler
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

    @property
    def is_exploration_phase(self):
        return self.initial_sampler is None or 0 < self.data.labels.sum() < self.data.labeled_size

    @property
    def is_initial_sampling_phase(self):
        return not self.is_exploration_phase

    def clear(self):
        self.data.clear(copy=True)
        self.active_learner.clear()
        self.__iter = 0

    def advance(self, new_idx, partial_labels, final_labels, user):
        metrics = {}

        metrics['phase'] = 'exploration' if self.is_exploration_phase else 'initial_sampling'

        # update partitions
        self.data.move_to_labeled(new_idx, partial_labels, final_labels, 'user')
        metrics['labeled_indexes'] = new_idx
        metrics['final_labels'] = list(final_labels)
        metrics['partial_labels'] = partial_labels.tolist()

        # get next to labels
        idx = self.initial_sampling_advance(user, metrics) if self.is_initial_sampling_phase else self.exploration_advance(metrics)

        # update iteration counter
        if metrics['phase'] == 'exploration':
            self.__iter += 1

        return idx, metrics

    def exploration_advance(self, metrics):
        # fit active learning model
        t0 = perf_counter()
        self._fit()
        metrics['fit_time'] = perf_counter() - t0

        # find next point to label
        t0 = perf_counter()
        idx, _ = self._get_next_point_to_label()
        metrics['get_next_time'] = perf_counter() - t0
        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        return idx

    def initial_sampling_advance(self, user, metrics):
        t0 = perf_counter()
        idx = self.initial_sampler(self.data, user)
        metrics['iter_time'] = perf_counter() - t0

        return idx

    def predict_labels(self):
        return self.active_learner.predict(self.data.data)

    def _fit(self):
        self.active_learner.fit_data(self.data)

    def _get_next_point_to_label(self):
        if self.data.unlabeled_size == 0:
            raise RuntimeError("The entire dataset has already been labeled!")

        return self.active_learner.next_points_to_label(self.data, self.subsampling)
