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

from typing import Any, TYPE_CHECKING, Union, Sequence, Optional

from .base import DualSpaceModelBase
from .model import PolytopeModel, FactorizedPolytopeModel, PolytopeModelBase
from ..active_learner import FactorizedActiveLearner

if TYPE_CHECKING:
    from ..active_learner import ActiveLearner


class DualSpaceModel(DualSpaceModelBase):
    """
    Dual Space Model algorithm: no factorization
    """
    def __init__(self, active_learner: ActiveLearner, mode: Union[str] = 'persist',
                 sample_unknown_proba: float = 0.5, tol: float = 1e-12):
        super().__init__(PolytopeModel(mode, tol), active_learner, sample_unknown_proba)


class FactorizedDualSpaceModel(DualSpaceModelBase, FactorizedActiveLearner):
    """
    Dual Space Model algorithm with factorization
    """
    def __init__(self, active_learner: ActiveLearner, sample_unknown_proba: float = 0.5,
                 partition: Optional[Sequence[Sequence[int]]] = None, mode: Union[str, Sequence[str]] = 'persist',
                 tol: float = 1e-12):
        pol = None if partition is None else FactorizedPolytopeModel(partition, mode, tol)
        super().__init__(pol, active_learner, sample_unknown_proba)
        self.__tol = tol

    def set_factorization_structure(self, **factorization_info: Any) -> None:
        partition, mode = factorization_info['partition'], factorization_info['mode']
        self.polytope_model: PolytopeModelBase = FactorizedPolytopeModel(partition, mode, self.__tol)
