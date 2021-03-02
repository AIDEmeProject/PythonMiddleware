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

import enum
from typing import Optional, TYPE_CHECKING

from ..utils import assert_positive_integer, process_callback, metric_logger
from ..utils.convergence import all_points_are_labeled

if TYPE_CHECKING:
    from . import LabeledSet, PartitionedDataset
    from .partitioned import IndexedDataset
    from ..active_learning import ActiveLearner
    from ..utils import InitialSampler, FunctionList, Callback, Convergence, Metrics


class Phase(enum.Enum):
    INITIAL = 'initial_sampling'
    EXPLORE = 'exploration'


class ExplorationManager:
    """
    Class for managing all aspects of the data exploration loop: initial sampling, model update, partition updates,
    callback computation, convergence detection, etc.
    """
    def __init__(self, data: PartitionedDataset, active_learner: ActiveLearner, subsampling: Optional[int],
                 initial_sampler: Optional[InitialSampler] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1,
                 convergence_criteria: FunctionList[Convergence] = None):
        """
        :param data: a PartitionedDataset instance
        :param active_learner: a ActiveLearner instance
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done.
        :param subsampling: sample size of unlabeled points when looking for the next point to label.
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        """
        assert_positive_integer(subsampling, 'subsampling', allow_none=True)
        assert_positive_integer(callback_skip, 'callback_skip')

        self.data = data
        self.active_learner = active_learner
        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip

        self.convergence_criteria = process_callback(convergence_criteria)
        self.convergence_criteria.append(all_points_are_labeled)

        self.__iters = 0

    @property
    def iters(self) -> int:
        """
        :return: How many iterations have gone so far. One iteration is considered to end when 'update()' is called
        """
        return self.__iters

    @property
    def phase(self) -> Phase:
        """
        :return: a string corresponding to the current iteration phase.
        """
        return Phase.EXPLORE if self.is_exploration_phase else Phase.INITIAL

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

    def clear(self) -> None:
        """
        Resets the internal state of all data structures and objects.
        """
        self.data.clear()
        self.active_learner.clear()
        self.__iters = 0

    @metric_logger.log_execution_time('fit_time')
    def update(self, labeled_set: LabeledSet) -> None:
        """
        Updates the Active Learning model and internal data structures. Call to this method marks the end of an "iteration".
        :param labeled_set: collection of new user labeled points
        """
        self.data.move_to_labeled(labeled_set)

        if self.is_exploration_phase:
            self.active_learner.fit_data(self.data)

        self.__iters += 1

    @metric_logger.log_execution_time('get_next_time')
    def get_next_to_label(self) -> Optional[IndexedDataset]:
        """
        :return: the next data points to be labeled
        """
        if self.data.unlabeled_size == 0:
            return None

        if self.is_initial_sampling_phase:
            return self.data.from_index(self.initial_sampler(self.data))

        return self.active_learner.next_points_to_label(self.data, self.subsampling)

    def converged(self) -> bool:
        """
        :return: whether the exploration is to be stopped or not
        """
        return any((criterion(self, metric_logger.get_metrics()) for criterion in self.convergence_criteria))

    def get_callback_metrics(self) -> Metrics:
        """
        :return: a dictionary of all iteration metrics. Callbacks are also included every 'callback_skip' iterations.
        """
        return self.__compute_callback_metrics() if self.__is_callback_computation_iter() else {}

    def __is_callback_computation_iter(self) -> bool:
        return (self.iters - 1) % self.callback_skip == 0 or self.converged()

    @metric_logger.log_execution_time('callback_time')
    def __compute_callback_metrics(self) -> Metrics:
        metrics: Metrics = {}

        for callback in self.callbacks:
            callback_metrics = callback(self.data, self.active_learner)

            if callback_metrics:
                metrics.update(callback_metrics)

        return metrics
