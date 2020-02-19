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

from typing import Optional, TYPE_CHECKING, Sequence, Union

from . import LabeledSet, ExplorationManager, PartitionedDataset
from ..utils import assert_positive_integer, process_callback

if TYPE_CHECKING:
    import numpy as np
    from ..active_learning import ActiveLearner
    from ..utils import Callback, Convergence, InitialSampler, FunctionList, Seed, RunType, RunsType


class PoolBasedExploration:
    def __init__(self, initial_sampler: Optional[InitialSampler] = None, subsampling: Optional[int] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1,
                 convergence_criteria: FunctionList[Convergence] = None):
        """
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done
        :param subsampling: sample size of unlabeled points when looking for the next point to label
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        """
        assert_positive_integer(subsampling, 'subsampling', allow_none=True)
        assert_positive_integer(callback_skip, 'callback_skip')

        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.convergence_criteria = process_callback(convergence_criteria)

    def run(self, data: np.ndarray, labeled_set: LabeledSet, active_learner: ActiveLearner, repeat: int = 1,
            seeds: Union[Seed, Sequence[Seed]] = None, copy: bool = True, return_generator: bool = True) -> RunsType:
        """
        Run the Active Learning model over data, for a given number of iterations.

        :param data: data matrix as a numpy array
        :param labeled_set: object containing the user labels, as a LabeledSet instance or array-like (no factorization in this case)
        :param active_learner: ActiveLearner instance to simulate
        :param repeat: number of times to repeat exploration
        :param seeds: list of random number generator seeds for each run. Set this if you wish for reproducible results.
        :param copy: whether to use a copy of the data matrix, avoiding changes to it.
        :param return_generator: whether to return a run and metrics as a generator. This way, you can get access to metrics
        as they are computed, not only when all runs are finished computing.
        :return: a list (or generator) of metrics collected after every iteration run. For each iteration we have a dictionary containing:
                - Labeled points (index, labels, partial_labels)
                - Timing measurements (fit, get next point, ...)
                - Any metrics returned by the callback function
        """
        seeds = self.__get_seed(seeds, repeat)

        if not isinstance(labeled_set, LabeledSet):
            labeled_set = LabeledSet(labeled_set)

        index = labeled_set.index
        if not copy:  # always copy labeled_set index since it will be changed in-place
            index = index.copy()

        data = PartitionedDataset(data, index, copy=copy)

        manager = ExplorationManager(
            data, active_learner, initial_sampler=self.initial_sampler, subsampling=self.subsampling,
            callback=self.callbacks, callback_skip=self.callback_skip,
            convergence_criteria=self.convergence_criteria
        )

        runs = (self._run(manager, labeled_set, seed) for seed in seeds)
        return runs if return_generator else [list(run) for run in runs]

    def _run(self, manager: ExplorationManager, labeled_set: LabeledSet, seed: Optional[int]) -> RunType:
        manager.clear(seed)

        converged, new_labeled_set = False, None
        while not converged:
            idx, metrics, converged = manager.advance(new_labeled_set)

            yield metrics

            new_labeled_set = labeled_set.get_index(idx)  # "User labeling"

    def __get_seed(self, seed: Union[Seed, Sequence[Seed]], repeat: int) -> Sequence[Seed]:
        if seed is None:
            seed = [None] * repeat
        elif isinstance(seed, int):
            seed = [seed]

        if len(seed) != repeat:
            raise ValueError("Expected {} seed values, but got {} instead.".format(repeat, len(seed)))

        return seed


class CommandLineExploration:
    """
    A class for running the exploration process on the command line.
    """

    def __init__(self, initial_sampler: Optional[InitialSampler] = None, subsampling: Optional[int] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1,
                 convergence_criteria: FunctionList[Convergence] = None):
        """
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done
        :param subsampling: sample size of unlabeled points when looking for the next point to label
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        """
        assert_positive_integer(subsampling, 'subsampling', allow_none=True)
        assert_positive_integer(callback_skip, 'callback_skip')

        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.convergence_criteria = process_callback(convergence_criteria)

    def run(self, X, active_learner: ActiveLearner) -> None:
        data = PartitionedDataset(X, copy=False)

        manager = ExplorationManager(
            data, active_learner, self.subsampling, self.initial_sampler,
            self.callbacks, self.callback_skip,
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
