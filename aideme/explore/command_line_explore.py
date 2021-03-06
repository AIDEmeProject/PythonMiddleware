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

from typing import Optional, TYPE_CHECKING

from . import LabeledSet, ExplorationManager, PartitionedDataset
from ..utils import assert_positive_integer, process_callback

if TYPE_CHECKING:
    from ..active_learning import ActiveLearner
    from ..utils import Callback, Convergence, InitialSampler, FunctionList


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

        while not manager.converged() and self._is_willing:
            data_to_label = manager.get_next_to_label()
            new_labeled_set = self._label(data_to_label.index, data_to_label.data)
            manager.update(new_labeled_set)

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
