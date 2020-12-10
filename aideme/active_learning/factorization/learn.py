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

from typing import TYPE_CHECKING

import numpy as np

from aideme.utils import assert_in_range

if TYPE_CHECKING:
    from .linear import LinearFactorizationLearner


def prune_irrelevant_subspaces(X: np.ndarray, learner: LinearFactorizationLearner, threshold: float = 0.9) -> LinearFactorizationLearner:
    assert_in_range(threshold, 'threshold', low=0, high=1)

    partial_probas = learner.partial_proba(X)
    relevant_rows = np.where(partial_probas.min(axis=0) < threshold)[0]

    pruned_learner = learner.copy()
    pruned_learner._weights = learner._weights[relevant_rows]
    pruned_learner._bias = learner._bias[relevant_rows]

    return pruned_learner


def compute_factorization_structure(X: np.ndarray, learner: LinearFactorizationLearner) -> np.ndarray:
    # remove irrelevant features
    importance_weights = np.abs(learner.weights)
    importance_weights /= importance_weights.sum(axis=1).reshape(-1, 1)
    return importance_weights > 1 / X.shape[1]
