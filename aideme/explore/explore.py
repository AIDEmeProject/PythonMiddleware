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

from typing import Optional, List, TYPE_CHECKING

from . import LabeledSet, ExplorationManager, PartitionedDataset
from ..utils import assert_positive_integer, process_callback

if TYPE_CHECKING:
    from ..active_learning import ActiveLearner
    from ..utils import Callback, Convergence, InitialSampler, FunctionList, Metrics


class PoolBasedExploration:
    def __init__(self, initial_sampler: Optional[InitialSampler] = None, subsampling: Optional[int] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1, print_callback_result: bool = False,
                 convergence_criteria: FunctionList[Convergence] = None):
        """
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done
        :param subsampling: sample size of unlabeled points when looking for the next point to label
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param print_callback_result: whether to print all callback metrics to stdout
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        """
        if subsampling is not None:
            assert_positive_integer(subsampling, 'subsampling')
        assert_positive_integer(callback_skip, 'callback_skip')

        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.print_callback_result = print_callback_result
        self.convergence_criteria = process_callback(convergence_criteria)

    def run(self, X, labeled_set, active_learner: ActiveLearner, repeat: int = 1) -> List[List[Metrics]]:
        """
        Run the Active Learning model over data, for a given number of iterations.

        :param X: data matrix as a numpy array
        :param labeled_set: object containing the user labels, as a LabeledSet instance or array-like (no factorization in this case)
        :param active_learner: ActiveLearner instance to simulate
        :param repeat: number of times to repeat exploration
        :return: a list of metrics collected after every iteration run. For each iteration we have a dictionary
        containing:
                - Labeled points (index, labels, partial_labels)
                - Timing measurements (fit, get next point, ...)
                - Any metrics returned by the callback function
        """
        if not isinstance(labeled_set, LabeledSet):
            labeled_set = LabeledSet(labeled_set)

        data = PartitionedDataset(X, labeled_set.index)

        manager = ExplorationManager(
            data, active_learner, initial_sampler=self.initial_sampler, subsampling=self.subsampling,
            callback=self.callbacks, callback_skip=self.callback_skip, print_callback_result=self.print_callback_result,
            convergence_criteria=self.convergence_criteria
        )

        return [self._run(manager, labeled_set) for _ in range(repeat)]

    def _run(self, manager: ExplorationManager, labeled_set: LabeledSet) -> List[Metrics]:
        manager.clear()

        converged, new_labeled_set = False, None

        iter_metrics = []
        while not converged:
            idx, metrics, converged = manager.advance(new_labeled_set)
            iter_metrics.append(metrics)

            new_labeled_set = labeled_set.get_index(idx)  # "User labeling"

        return iter_metrics


class CommandLineExploration:
    """
    A class for running the exploration process on the command line.
    """

    def __init__(self, initial_sampler: Optional[InitialSampler] = None, subsampling: Optional[int] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1, print_callback_result: bool = False,
                 convergence_criteria: FunctionList[Convergence] = None):
        """
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done
        :param subsampling: sample size of unlabeled points when looking for the next point to label
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param print_callback_result: whether to print all callback metrics to stdout
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        """
        if subsampling is not None:
            assert_positive_integer(subsampling, 'subsampling')
        assert_positive_integer(callback_skip, 'callback_skip')

        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.print_callback_result = print_callback_result
        self.convergence_criteria = process_callback(convergence_criteria)

    def run(self, X, active_learner: ActiveLearner) -> None:
        data = PartitionedDataset(X, copy=False)

        manager = ExplorationManager(
            data, active_learner, self.subsampling, self.initial_sampler,
            self.callbacks, self.callback_skip, self.print_callback_result,
            self.convergence_criteria
        )

        print('Welcome to the manual exploration process. \n')

        idx, metrics, converged = manager.advance()

        while not converged and self._is_willing:
            labels = self._label(idx, X[idx])
            idx, metrics, converged = manager.advance(labels)

    @property
    def _is_willing(self) -> bool:
        val = input("Continue (y/n): ")
        while val not in ['y', 'n']:
            val = input("Continue (y/n): ")

        return True if val == 'y' else False

    def _label(self, idx, pts) -> LabeledSet:
        is_valid, labels = self.__is_valid_input(pts)
        while not is_valid:
            is_valid, labels = self.__is_valid_input(pts)
        return LabeledSet(labels, index=idx)

    @staticmethod
    def __is_valid_input(pts):
        s = input("Give the labels for the following points: {}\n".format(pts.tolist()))
        expected_size = len(pts)

        if not set(s).issubset({' ', '0', '1'}):
            print("Invalid character in labels. Only 0, 1 and ' ' are permitted.\n")
            return False, None

        vals = s.split()
        if len(vals) != expected_size:
            print('Incorrect number of labels: got {} but expected {}\n'.format(len(vals), expected_size))
            return False, None

        print()
        return True, [int(x) for x in vals]
