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

from typing import Union, Sequence, TypeVar, Callable, Dict, Any, Optional, Tuple

import numpy as np

from aideme.active_learning.active_learner import ActiveLearner  # Use full import path to avoid circular dependency
from aideme.explore.manager import ExplorationManager
from aideme.explore.partitioned import PartitionedDataset
from aideme.explore.labeledset import LabeledSet


T = TypeVar('T')
FunctionList = Union[None, T, Sequence[T]]

Metrics = Dict[str, Any]
Callback = Callable[[PartitionedDataset, ActiveLearner], Metrics]
Convergence = Callable[[ExplorationManager, Metrics], bool]

InitialSampler = Callable[[PartitionedDataset], Sequence]
Seed = Optional[int]

Partition = Sequence[Union[slice, Sequence[int]]]
HyperPlane = Tuple[float, np.ndarray]

NoiseInjector = Callable[[int, LabeledSet], LabeledSet]
