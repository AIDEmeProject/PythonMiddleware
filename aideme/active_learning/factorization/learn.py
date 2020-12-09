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

from typing import TYPE_CHECKING, Union, List

import numpy as np

from aideme.utils import assert_in_range
from .linear import LinearFactorizationLearner

if TYPE_CHECKING:
    from .optimization import OptimizationAlgorithm


def compute_factorization_structure(X: np.ndarray, y: np.ndarray, factorization: Union[int, List[List[int]]], optimizer: OptimizationAlgorithm, repeat: int = 1,
                                    threshold: float = 0.9, l2_penalty: float = 0, l2_sqrt_penalty: float = 1e-4) -> np.ndarray:
    assert_in_range(threshold, 'threshold', low=0, high=1)

    # fit factorized classifier
    learner = LinearFactorizationLearner(optimizer, l2_penalty=l2_penalty, l2_sqrt_penalty=l2_sqrt_penalty)
    learner.fit(X, y, factorization, repeat)

    # prune irrelevant subspaces
    partial_probas = learner.partial_proba(X)
    relevant_rows = np.where(partial_probas.min(axis=0) < threshold)[0]
    pruned_weights = learner._weights[relevant_rows]

    # remove irrelevant features
    importance_weights = np.abs(pruned_weights)
    importance_weights /= importance_weights.sum(axis=1).reshape(-1, 1)
    return importance_weights > 1 / X.shape[1]
