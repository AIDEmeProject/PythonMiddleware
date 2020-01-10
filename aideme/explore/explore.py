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

from .labeledset import LabeledSet
from .partitioned import PartitionedDataset
from ..utils.validation import assert_positive_integer, process_callback
from ..utils.convergence import all_points_are_labeled


class PoolBasedExploration:
    def __init__(self, initial_sampler=None, subsampling=math.inf,
                 callback=None, callback_skip=1, print_callback_result=False,
                 convergence_criteria=None):
        """
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done.
        :param callback: a callback function to be called at the end of every iteration. It must have the
        following signature:
                          callback(manager), where manager is a ExplorationManager object

        The callback can optionally return a dictionary containing new metrics to be included.
        :param callback_skip: compute callback every callback_skip iterations only.
        """
        assert_positive_integer(subsampling, 'subsampling', allow_inf=True)
        assert_positive_integer(callback_skip, 'callback_skip')

        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.print_callback_result = print_callback_result
        self.convergence_criteria = process_callback(convergence_criteria)

    def run(self, X, labeled_set, active_learner, repeat=1):
        """
        Run the Active Learning model over data, for a given number of iterations.

        :param X: data matrix
        :param labeled_set: object containing the user labels, as a LabeledSet instance or 1D numpy array (no factorization in this case)
        :param active_learner: ActiveLearner instance to simulate
        :param repeat: number of times to repeat exploration
        :return: a list of metrics collected after every iteration run. For each iteration we have a dictionary
        containing:
                - Labeled points (index, labels, partial_labels)
                - Timing measurements (fit, get next point, ...)
                - Any metrics returned by the callback function
        """
        data = PartitionedDataset(X, copy=True)

        if not isinstance(labeled_set, LabeledSet):
            labeled_set = LabeledSet(labeled_set)

        iteration = ExplorationManager(
            data, active_learner, initial_sampler=self.initial_sampler, subsampling=self.subsampling,
            callback=self.callbacks, callback_skip=self.callback_skip, print_callback_result=self.print_callback_result,
            convergence_criteria=self.convergence_criteria
        )

        return [self._run(iteration, labeled_set) for _ in range(repeat)]

    def _run(self, iteration, labeled_set):
        iteration.clear()

        idx, metrics, converged = iteration.advance()

        iter_metrics = [metrics]
        while not converged:
            idx, metrics, converged = iteration.advance(labeled_set[idx])
            iter_metrics.append(metrics)

        return iter_metrics



class ExplorationManager:
    """
    Class for managing all aspects of the data exploration loop: initial sampling, model update, partition updates,
    callback computation, convergence detection, etc.
    """
    def __init__(self, data, active_learner, initial_sampler=None, subsampling=math.inf,
                 callback=None, callback_skip=1, print_callback_result=False,
                 convergence_criteria=None):
        assert_positive_integer(subsampling, 'subsampling', allow_inf=True)

        self.data = data
        self.active_learner = active_learner
        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.print_callback_result = print_callback_result

        self.convergence_criteria = process_callback(convergence_criteria)
        self.convergence_criteria.append(all_points_are_labeled)

        self.__initial_sampling_iters = 0
        self.__exploration_iters = 0

    @property
    def initial_sampling_iters(self):
        """
        :return: How many initial sampling iterations have gone so far.
        """
        return self.__initial_sampling_iters

    @property
    def exploration_iters(self):
        """
        :return: How many exploration iterations have gone so far.
        """
        return self.__exploration_iters

    @property
    def phase(self):
        """
        :return: a string corresponding to the current iteration phase.
        """
        return 'exploration' if self.is_exploration_phase else 'initial_sampling' if self.__initial_sampling_iters > 0 else 'begin'

    @property
    def is_exploration_phase(self):
        """
        :return: Whether we are at exploration phase (i.e. initial sampling phase is over)
        """
        return self.initial_sampler is None or 0 < self.data.labels.sum() < self.data.labeled_size

    @property
    def is_initial_sampling_phase(self):
        """
        :return: Whether we are at initial sampling phase. Initial sampling ends only after 1 negative and 1 positive
        points have been labeled by the user.
        """
        return not self.is_exploration_phase

    def clear(self):
        """
        Resets the iteration state.
        """
        self.data.clear(copy=True)
        self.active_learner.clear()
        self.__initial_sampling_iters = 0
        self.__exploration_iters = 0

    def advance(self, labeled_set=None):
        """
        Updates current model with new labeled data and computes new point to be labeled.
        :param labeled_set: a LabeledSet instance containing the new labeled points by the user
        :return: index of next point to be labeled, metrics
        """
        metrics = {'phase': self.phase}  # from which phase the new labeled points have been computed

        self.__update_partitions(labeled_set, metrics)

        idx = self.__initial_sampling_advance(metrics) if self.is_initial_sampling_phase else self.__exploration_advance(metrics)

        return idx, metrics, self.__converged(metrics)

    def __update_partitions(self, labeled_set, metrics):
        if labeled_set is not None:
            self.data.move_to_labeled(labeled_set, 'user')
            metrics.update(labeled_set.asdict())

    def __initial_sampling_advance(self, metrics):
        t0 = perf_counter()
        idx = self.initial_sampler(self.data)
        metrics['iter_time'] = perf_counter() - t0

        self.__initial_sampling_iters += 1

        return idx

    def __exploration_advance(self, metrics):
        # fit active learning model
        t0 = perf_counter()
        self.active_learner.fit_data(self.data)
        metrics['fit_time'] = perf_counter() - t0

        # find next point to label
        t0 = perf_counter()
        idx, _ = self.active_learner.next_points_to_label(self.data, self.subsampling)
        metrics['get_next_time'] = perf_counter() - t0
        metrics['iter_time'] = metrics['get_next_time'] + metrics['fit_time']

        self.__exploration_iters += 1

        if self.__is_callback_computation_iter(metrics):
            metrics.update(self.__get_callback_metrics())

        return idx

    def __converged(self, metrics):
        return any((criterion(self, metrics) for criterion in self.convergence_criteria))

    def __is_callback_computation_iter(self, metrics):
        return (self.exploration_iters - 1) % self.callback_skip == 0 or self.__converged(metrics)

    def __get_callback_metrics(self):
        metrics = {}

        for callback in self.callbacks:
            callback_metrics = callback(self.data, self.active_learner)

            if callback_metrics:
                metrics.update(callback_metrics)

        if self.print_callback_result:
            scores_str = ', '.join((k + ': ' + str(v) for k, v in metrics.items()))
            print('iter: {0}, {1}'.format(self.exploration_iters, scores_str))

        return metrics
