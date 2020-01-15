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

from sklearn.utils import check_random_state

from .utils import assert_positive_integer


if TYPE_CHECKING:
    from .explore import LabeledSet
    from .utils import  Index, RandomStateType, InitialSampler


__all__ = ['stratified_sampler', 'fixed_sampler', 'random_sampler']


def stratified_sampler(labeled_set: LabeledSet, pos: int = 1, neg: int = 1, neg_in_all_subspaces: bool = False, random_state: RandomStateType = None) -> InitialSampler:
    """
    Binary stratified sampling method. Randomly selects a given number of positive and negative points from an array
    of labels.

    :param pos: Number of positive points to sample. Must be non-negative.
    :param neg: Number of negative points to sample. Must be non-negative.
    """
    assert_positive_integer(pos, 'pos')
    assert_positive_integer(neg, 'neg')

    pos_mask = (labeled_set.labels == 1)
    neg_mask = (labeled_set.partial.max(axis=1) == 0) if neg_in_all_subspaces else ~pos_mask

    pos_idx, neg_idx = labeled_set.index[pos_mask], labeled_set.index[neg_mask]
    rng = check_random_state(random_state)

    def sampler(data) -> Index:
        pos_samples = rng.choice(pos_idx, size=pos, replace=False)
        neg_samples = rng.choice(neg_idx, size=neg, replace=False)

        return list(pos_samples) + list(neg_samples)

    return sampler


def fixed_sampler(indexes: Index) -> InitialSampler:
    """
    Dummy sampler which returns a specified selection of indexes.
    """
    return lambda data: indexes


def random_sampler(sample_size: int) -> InitialSampler:
    """
    Samples a random batch of unlabeled points.
    """
    assert_positive_integer(sample_size, 'sample_size')
    return lambda data: data.sample_unlabeled(sample_size)[0]
