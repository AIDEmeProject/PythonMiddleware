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
from typing import Optional, List, TYPE_CHECKING, Sequence, Union, Generator

import numpy as np

from . import LabeledSet, ExplorationManager, PartitionedDataset
from ..utils import assert_positive_integer, process_callback

if TYPE_CHECKING:
    from ..active_learning import ActiveLearner
    from ..utils import Callback, Convergence, InitialSampler, FunctionList, Metrics, Seed, NoiseInjector
    RunType = Generator[Metrics, None, None]
    RunsType = Union[List[List[Metrics]], Generator[RunType, None, None]]


class PoolBasedExploration:
    def __init__(self, initial_sampler: Optional[InitialSampler] = None, subsampling: Optional[int] = None,
                 callback: FunctionList[Callback] = None, callback_skip: int = 1,
                 convergence_criteria: FunctionList[Convergence] = None, noise_injector: Optional[NoiseInjector] = None):
        """
        :param initial_sampler: InitialSampler object. If None, no initial sampling will be done
        :param subsampling: sample size of unlabeled points when looking for the next point to label
        :param callback: a list of callback functions. For more info, check utils/metrics.py
        :param callback_skip: compute callback every callback_skip iterations
        :param convergence_criteria: a list of convergence criterias. For more info, check utils/convergence.py
        :param noise_injector: a function for injecting labeling noise. For more info, check utils/noise.py
        """
        assert_positive_integer(subsampling, 'subsampling', allow_none=True)
        assert_positive_integer(callback_skip, 'callback_skip')

        self.initial_sampler = initial_sampler
        self.subsampling = subsampling

        self.callbacks = process_callback(callback)
        self.callback_skip = callback_skip
        self.convergence_criteria = process_callback(convergence_criteria)
        self.noise_injector = noise_injector

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
        manager.clear()
        self.__set_random_state(seed)

        while not manager.converged():
            metrics = {'phase': manager.phase}

            idx = manager.get_next_to_label()

            new_labeled_set = labeled_set.get_index(idx)  # "User labeling"
            metrics.update(new_labeled_set.asdict())

            if self.noise_injector and manager.is_exploration_phase:
                new_labeled_set = self.noise_injector(new_labeled_set)
                metrics.update(new_labeled_set.asdict(noisy=True))

            manager.update(new_labeled_set)

            metrics.update(manager.get_metrics())
            yield metrics

    @staticmethod
    def __get_seed(seed: Union[Seed, Sequence[Seed]], repeat: int) -> Sequence[Seed]:
        if seed is None:
            seed = [None] * repeat
        elif isinstance(seed, int):
            seed = [seed]

        if len(seed) != repeat:
            raise ValueError("Expected {} seed values, but got {} instead.".format(repeat, len(seed)))

        return seed

    @staticmethod
    def __set_random_state(seed: Optional[int] = None) -> None:
        np.random.seed(seed)  # Seeds numpy's internal RNG
        random.seed(seed)  # Seeds python's internal RNG

    def __inject_noise(self, labeled_set: LabeledSet) -> LabeledSet:
        return labeled_set if self.noise_injector is None else self.noise_injector(labeled_set)
