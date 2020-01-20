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
from __future__ import annotations

import random
from time import perf_counter
from typing import Tuple, Optional, TYPE_CHECKING, List, Sequence

import numpy as np

from ..utils import assert_positive_integer, process_callback
from ..utils.convergence import all_points_are_labeled

if TYPE_CHECKING:
    from . import LabeledSet, PartitionedDataset
    from ..active_learning import ActiveLearner
    from ..utils import InitialSampler, FunctionList, Callback, Convergence, Metrics


class ExplorationManager:
    """
    Class for managing all aspects of the data exploration loop: initial sampling, model update, partition updates,
    callback computation, convergence detection, etc.
    """
    def __init__(self, data: PartitionedDataset, active_learner: ActiveLearner, subsampling: Optional[int],
                 initial_sampler: Optional[InitialSampler] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1, print_callback_result: bool = False,
                 convergence_criteria: FunctionList[Convergence] = None):
        """
        :param data: a PartitionedDataset instance
        :param active_learner: a ActiveLearner instance
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done.
        :param subsampling: sample size of unlabeled points when looking for the next point to label.
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param print_callback_result: whether to print all callback metrics to stdout
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        """
        if subsampling is not None:
            assert_positive_integer(subsampling, 'subsampling')
        assert_positive_integer(callback_skip, 'callback_skip')

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
    def initial_sampling_iters(self) -> int:
        """
        :return: How many initial sampling iterations have gone so far.
        """
        return self.__initial_sampling_iters

    @property
    def exploration_iters(self) -> int:
        """
        :return: How many exploration iterations have gone so far.
        """
        return self.__exploration_iters

    @property
    def phase(self) -> str:
        """
        :return: a string corresponding to the current iteration phase.
        """
        return 'exploration' if self.is_exploration_phase else 'initial_sampling' if self.__initial_sampling_iters > 0 else 'begin'

    @property
    def is_exploration_phase(self) -> bool:
        """
        :return: Whether we are at exploration phase (i.e. initial sampling phase is over)
        """
        return self.initial_sampler is None or self.data.labeled_set.has_positive_and_negative_labels()

    @property
    def is_initial_sampling_phase(self) -> bool:
        """
        :return: Whether we are at initial sampling phase. Initial sampling ends only after 1 negative and 1 positive
        points have been labeled by the user.
        """
        return not self.is_exploration_phase

    def clear(self, seed: Optional[int] = None) -> None:
        """
        Resets the internal state of all data structures and objects.
        :param seed: seed for numpy's internal RNG. Set this if you need reproducible results.
        """
        self.__set_random_state(seed)

        self.data.clear()
        self.active_learner.clear()
        self.__initial_sampling_iters = 0
        self.__exploration_iters = 0

    def __set_random_state(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)  # Seeds numpy's internal RNG
        random.seed(seed)  # Seeds python's internal RNG

    def advance(self, labeled_set: Optional[LabeledSet] = None) -> Tuple[Sequence, Metrics, bool]:
        """
        Updates current model with new labeled data and computes new point to be labeled.
        :param labeled_set: a LabeledSet instance containing the new labeled points by the user
        :return: index of next point to be labeled, metrics
        """
        metrics = {'phase': self.phase}  # from which phase the new labeled points have been computed

        self.__update_partitions(labeled_set, metrics)

        idx = self.__initial_sampling_advance(metrics) if self.is_initial_sampling_phase else self.__exploration_advance(metrics)

        return idx, metrics, self.__converged(metrics)

    def __update_partitions(self, labeled_set: Optional[LabeledSet], metrics: Metrics) -> None:
        if labeled_set is not None:
            self.data.move_to_labeled(labeled_set)
            metrics.update(labeled_set.asdict())

    def __initial_sampling_advance(self, metrics: Metrics) -> Sequence:
        if self.initial_sampler is None:
            raise RuntimeError("No initial sampler was provided.")

        t0 = perf_counter()
        idx = self.initial_sampler(self.data)
        metrics['iter_time'] = perf_counter() - t0

        self.__initial_sampling_iters += 1

        return idx

    def __exploration_advance(self, metrics: Metrics) -> Sequence:
        # fit active learning model
        t0 = perf_counter()
        self.active_learner.fit_data(self.data)
        metrics['fit_time'] = perf_counter() - t0

        # find next point to label
        idx: List = []
        if self.data.unlabeled_size > 0:
            t0 = perf_counter()
            idx = self.active_learner.next_points_to_label(self.data, self.subsampling).index
            metrics['get_next_time'] = perf_counter() - t0

        metrics['iter_time'] = metrics['fit_time'] + metrics.get('get_next_time', 0)

        self.__exploration_iters += 1

        if self.__is_callback_computation_iter(metrics):
            t0 = perf_counter()
            metrics.update(self.__get_callback_metrics())
            metrics['callback_time'] = perf_counter() - t0

        return idx

    def __converged(self, metrics: Metrics) -> bool:
        return any((criterion(self, metrics) for criterion in self.convergence_criteria))

    def __is_callback_computation_iter(self, metrics: Metrics) -> bool:
        return (self.exploration_iters - 1) % self.callback_skip == 0 or self.__converged(metrics)

    def __get_callback_metrics(self) -> Metrics:
        metrics: Metrics = {}

        for callback in self.callbacks:
            callback_metrics = callback(self.data, self.active_learner)

            if callback_metrics:
                metrics.update(callback_metrics)

        if self.print_callback_result:
            scores_str = ', '.join((k + ': ' + str(v) for k, v in metrics.items()))
            print('iter: {0}, {1}'.format(self.exploration_iters, scores_str))

        return metrics
