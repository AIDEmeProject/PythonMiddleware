#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.
from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING, Sequence, Union, Generator

from . import LabeledSet, ExplorationManager, PartitionedDataset
from ..utils import assert_positive_integer, process_callback, metric_logger
from ..utils.random import set_random_state

if TYPE_CHECKING:
    import numpy as np
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
        set_random_state(seed)

        manager.clear()

        while not manager.converged():
            metric_logger.flush()  # avoid overlapping metrics between iterations
            metric_logger.log_metric('phase', manager.phase.value)

            self.__run_single_iter(labeled_set, manager)

            metric_logger.log_metrics(manager.get_callback_metrics())
            yield metric_logger.get_metrics()

        metric_logger.flush()

    @metric_logger.log_execution_time('iter_time')
    def __run_single_iter(self, labeled_set: LabeledSet, manager: ExplorationManager) -> None:
        data = manager.get_next_to_label()

        user_labels = labeled_set.get_index(data.index)  # 'User labeling'
        metric_logger.log_metrics(user_labels.asdict())

        if self.noise_injector and manager.is_exploration_phase:
            user_labels = self.noise_injector(manager.iters, user_labels)
            metric_logger.log_metrics(user_labels.asdict(noisy=True))

        manager.update(user_labels)

    @staticmethod
    def __get_seed(seed: Union[Seed, Sequence[Seed]], repeat: int) -> Sequence[Seed]:
        if seed is None:
            seed = [None] * repeat
        elif isinstance(seed, int):
            seed = [seed]

        if len(seed) != repeat:
            raise ValueError("Expected {} seed values, but got {} instead.".format(repeat, len(seed)))

        return seed
