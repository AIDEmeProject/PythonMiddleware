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

from aideme.utils import assert_in_range, assert_positive
from .linear import LinearFactorizationLearner
from .optimization import Adam, FISTA
from .penalty import SparseGroupLassoPenalty

if TYPE_CHECKING:
    from aideme.explore import PartitionedDataset


def prune_irrelevant_subspaces(X: np.ndarray, learner: LinearFactorizationLearner,
                               threshold: float = 0.99, tol: float = 1e-8) -> LinearFactorizationLearner:
    assert_in_range(threshold, 'threshold', low=0, high=1)
    assert_positive(tol, 'tol')

    is_relevant = np.logical_and(
        learner.partial_proba(X).min(axis=0) < threshold,
        np.linalg.norm(learner._weights, axis=1) > tol
    )
    relevant_rows = np.where(is_relevant)[0]

    pruned_learner = learner.copy()
    pruned_learner._weights = learner._weights[relevant_rows]
    pruned_learner._bias = learner._bias[relevant_rows]

    return pruned_learner


def compute_relevant_attributes(learner: LinearFactorizationLearner) -> np.ndarray:
    groups = learner.feature_groups
    importance_weights = np.abs(learner._weights)
    importance_weights /= importance_weights.sum(axis=1).reshape(-1, 1)
    threshold = 1 / importance_weights.shape[1]

    if groups is not None:
        threshold = 1 / len(groups)
        for g in groups:
            importance_weights[:, g] = importance_weights[:, g].sum(axis=1).reshape(-1, 1)

    return importance_weights > threshold


def compute_factorization_and_partial_labels(dataset: PartitionedDataset, linear_model: LinearFactorizationLearner,
                                             fista_step_size: float = 5, max_iter: int = 2500, l1_penalty: float = 1e-4, l2_sqrt_penalty: float = 1e-4):
    global_opt_params = {'batch_size': None, 'adapt_step_size': False, 'max_iter': max_iter}
    X, y = dataset.training_set()

    # Step 1: refine the current FLM into a sparse model
    optimizer = FISTA(step_size=fista_step_size, **global_opt_params)
    penalty = SparseGroupLassoPenalty(l1_penalty=l1_penalty, l2_sqrt_penalty=l2_sqrt_penalty)
    refining_model = LinearFactorizationLearner(optimizer, penalty)
    refining_model.fit(X, y, linear_model.num_subspaces, x0=linear_model.weight_matrix)

    # Step 2: prune irrelevant subspaces and attributes
    pruned = prune_irrelevant_subspaces(dataset.data, refining_model)
    relevant_attrs = compute_relevant_attributes(pruned)

    # Step 3: train a model with fixed irrelevant attributes removed (set irrelevant weights to zero)
    subspaces = [list(np.where(s)[0]) for s in relevant_attrs]
    optimizer = Adam(step_size=0.1, **global_opt_params)
    sparse_model = LinearFactorizationLearner(optimizer=optimizer)
    sparse_model.fit(X, y, subspaces, x0=pruned.weight_matrix)

    # Step 4: compute factorization structure and partial labels
    factorization = compute_factorization(relevant_attrs)
    y_partial = __compute_partial_labels(subspaces, factorization, sparse_model.partial_proba(X))

    return factorization, y_partial


def compute_factorization(relevant_attrs):
    unique_subspaces = [list(np.where(s)[0]) for s in np.unique(relevant_attrs, axis=0)]
    factorization = []
    for i, s in enumerate(unique_subspaces):
        attrs = set(s)
        if not any(attrs.issubset(r) for j, r in enumerate(unique_subspaces) if i != j):
            factorization.append(s)
    return sorted(factorization)


def __compute_partial_labels(subspaces, factorization, p):
    pos = []
    for s in subspaces:
        res = []
        attrs = set(s)
        for i, f in enumerate(factorization):
            if attrs.issubset(f):
                res.append(i)
        pos.append(res)

    ps = np.ones((len(p), len(factorization)))
    for i in range(ps.shape[1]):
        for j, (idx, s) in enumerate(zip(pos, subspaces)):
            if i in idx:
                ps[:, i] *= np.power(p[:, j], 1 / len(idx))

    return (ps > 0.5).astype('float')
