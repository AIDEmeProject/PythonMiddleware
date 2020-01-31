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

NoiseInjector = Callable[[LabeledSet], LabeledSet]
